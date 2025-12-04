import asyncio
import argparse
import sys
import os
import json
from typing import Optional
from dotenv import load_dotenv
import hashlib

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate models on affine environments',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-envs', nargs="+", type=str, default=[],
    )
    # Mode selection: either uid or (model + base-url)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--model',
                           help='Model name (use with --base-url)')
    parser.add_argument('--base-url',
                       help='Model service URL (required with --model)')
    parser.add_argument('--concurrency', type=int, default=8,
                       help='Number of concurrent evaluations')
    parser.add_argument('--is-gen', action='store_true', default=False)
    args = parser.parse_args()

    # Validation
    if args.model and not args.base_url:
        parser.error('--base-url is required when using --model')

    return args

def count_rounds(conversation):
    return len(conversation[3:]) / 2

def format_messages(conversation):
    messages = []
    for turn in conversation[:3]:
        messages.append({
            "role": turn["role"],
            "content": turn["content"],
            "loss": False
        })
    for turn in conversation[3:-1]:
        messages.append({
            "role": turn["role"],
            "content": turn["content"],
            "loss": True if turn["role"] == "assistant" else False
        })
    return messages

def normalize_reward(reward, min_reward, max_reward):
    return (reward - min_reward) / (max_reward - min_reward)

def save_to_jsonl(data, file_path):
    with open(file_path, 'a') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def calculate_seed(env_name, task_id):
    seed_input = f"{env_name}:{task_id}"
    hash_bytes = hashlib.sha256(seed_input.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')

async def evaluate_with_model(env_name, env_instance, model: str, base_url: str, task_id: Optional[int], is_gen: bool=True):
    """Evaluate using direct model endpoint"""
        
    eval_kwargs = {
        'task_id': task_id,
        'model': model,
        'base_url': base_url,
        'temperature': 0.7,
        'seed': calculate_seed(f"agentgym:{env_name}", task_id)
    }
    
    try:
        result = await env_instance.evaluate(**eval_kwargs)

        if not is_gen:
            eval_result = {
                "idx": task_id,
                "reward": result.score
            }
        else:
            threshold = -0.1 if env_name == "sciworld" else 0.0
            try:
                eval_result = {
                    "idx": task_id,
                    "reward": result.score,
                    "rounds": count_rounds(result.extra['conversation']),
                    "messages": format_messages(result.extra['conversation']) if result.score > threshold else []
                }
                print(f"✓ Task {task_id} completed with score: {result.score}")
            except Exception as e:
                eval_result = {
                    "idx": task_id,
                    "reward": -100,
                    "rounds": 100,
                    "messages": []
                }
                print(f"✗ Task {task_id} failed: {e}")
        return eval_result
    except Exception as e:
        print(f"✗ Task {task_id} failed: {e}")
        return {
            "idx": task_id,
            "error": str(e)
        }

async def run_batch(env_name, env_instance, model, base_url, task_ids, is_gen):
    """Run a batch of evaluations concurrently"""
    tasks = [
        evaluate_with_model(env_name, env_instance, model, base_url, task_id, is_gen)
        for task_id in task_ids
    ]
    return await asyncio.gather(*tasks)

async def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Import affine AFTER argparse to avoid bittensor hijacking
    import affine as af

    # Map environment names to actual classes
    ENVIRONMENTS = {
        'alfworld': af.ALFWORLD,
        'webshop': af.WEBSHOP,
        'babyai': af.BABYAI,
        'sciworld': af.SCIWORLD,
        'textcraft': af.TEXTCRAFT,
    }

    ENVIRONMENTS_TO_SAMPLE_RANGE = {
        'alfworld': (0, 2500),
        'webshop': (0, 500),
        'babyai': (0, 500),
        'sciworld': (0, 2500),
        'textcraft': (0, 582),
    }

    MIN_MAX_REWARD = {
        "sciworld": (-100, 100)
    }
    
    # Set fake API key for local model testing (required by Docker env)
    if args.model and not os.getenv("CHUTES_API_KEY"):
        os.environ["CHUTES_API_KEY"] = "fake-test-key-for-local-testing"

    model_name = args.model.replace('/', '_')
    
    for env in args.envs:
        try:
            # Create environment instance
            print(f"\nLoading {env} environment...")
            env_class = ENVIRONMENTS[env]
            env_instance = env_class()
            print("✓ Environment loaded")
            
            # Run evaluation
            sample_range = ENVIRONMENTS_TO_SAMPLE_RANGE[env]
            print(f"\nStarting evaluation with {args.concurrency} concurrent tasks...")

            results = []
            output_file = f'{env}_{model_name}.jsonl'
            
            # Process in batches based on concurrency level
            for i in range(sample_range[0], sample_range[1], args.concurrency):
                batch_ids = list(range(i, min(i + args.concurrency, sample_range[1])))
                print(f"\nProcessing batch: tasks {batch_ids[0]} to {batch_ids[-1]}")
                
                batch_results = await run_batch(
                    env,
                    env_instance, 
                    args.model, 
                    args.base_url, 
                    batch_ids,
                    args.is_gen
                )
                results.extend(batch_results)
                
                print(f"Batch completed. Progress: {batch_ids[-1]+1-sample_range[0]}/{sample_range[1]-sample_range[0]}")

                if len(results) >= 32 and args.is_gen:
                    save_to_jsonl(results, output_file)
                    results = []

            if results and args.is_gen:
                save_to_jsonl(results, output_file)

            # Calculate and display average score
            if not args.is_gen:
                min_reward, max_reward = MIN_MAX_REWARD.get(env, (0, 1))
                rewards = [normalize_reward(result['reward'], min_reward, max_reward) for result in results if 'error' not in result]
                avg_score = sum(rewards) / len(rewards)
                print(f"Average score for {env}: {avg_score:.4f}")
                summary = {
                    "env": env,
                    "sample_range": sample_range,
                    "avg_score": avg_score,
                    "results": results
                }
                with open(f'summary_{env}_{model_name}.json', 'w') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        finally:
            print(f"Cleaning up {env} environment...")
            await env_instance._env.cleanup()
            print("✓ Environment cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
