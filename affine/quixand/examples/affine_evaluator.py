#!/usr/bin/env python3

import os
import json
import time
import logging
import argparse
from affine import quixand as qs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# if not os.getenv("CHUTES_API_KEY"):
#     logger.warning("CHUTES_API_KEY is not set. Proxy endpoints may not work correctly.")
#     exit(0)
    
SGLANG_PORT = 10000
MODEL = "openai/gpt-oss-20b"
MODEL_NAME = "_".join(MODEL.split("/")) 

def test_evaluator_endpoint(sandbox):
    ids = []
    rewards = []
    successes = []
    experiences = []
    
    for testcase_id in range(200):
        evaluation_request = {
            "model": MODEL,
            "ids": [testcase_id], # testcase ids
            "max_round": 10,
            "base_url": f"http://localhost:{SGLANG_PORT}/v1",
            "timeout": 600,
            "temperature": 0.7,
        }
        
        print(f"Problem ids: {evaluation_request['ids']}\n")
        
        start_time = time.time()
        try:
            response = sandbox.proxy.evaluator(**evaluation_request, _timeout=600)
            print(f"Response: {response}")
            elapsed = time.time() - start_time

            print(f"Evaluation completed in {elapsed:.2f} seconds\n")
            
            # Display results
            print("=== Evaluation Results ===")
            print(f"Task: {response['task_name']}")
            print(f"Average Score: {response['total_score']:.3f}")
            print(f"Success Rate: {response['success_rate']:.3f}")
            print(f"Time Taken: {response['time_taken']:.2f}s")
            
            # print("\n=== Detailed Results ===")
            for detail in response['details']:
                
                ids.append(detail['id'])
                rewards.append(detail['reward'])
                successes.append(detail['success'])
                try:
                    experiences.append(detail['experiences'])
                except:
                    experiences.append([])

        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            
    return ids, rewards, successes, experiences


def main():
    AVAILABLE_ENVS = [
        "sat",
        "abd",
        "ded"
    ]
    
    parser = argparse.ArgumentParser(description='Run AgentGym evaluator for specified environment')
    parser.add_argument(
        '-envs',
        nargs="+",
        type=str,
        default=[],
        help=f'Environment name to evaluate. Available options: {", ".join(AVAILABLE_ENVS)}'
    )
    
    args = parser.parse_args()
    env_names = args.envs
    
    for env_name in env_names:
        print(f"Building Docker image for {env_name} environment...")
        sandbox = qs.get_sandbox(f"affine:{env_name}")
        print(f"Container ID: {sandbox.container_id[:12]}\n")

        try:
            ids, rewards, successes, experiences = test_evaluator_endpoint(sandbox)

            if ids and rewards and successes:
                print("\n=== Test Summary ===")
                print("✓ Evaluator endpoint is working correctly!")
                print(f"Successfully evaluated {len(ids)} examples")
            else:
                print("\n=== Test Summary ===")
                print("✗ Evaluator endpoint test failed")
                print("Please check the error messages above")
                
            # Save results to a JSON file
            results = {
                "rewards": rewards,
                "successes": successes,
                "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
                "success_rate": sum(successes) / len(successes) if successes else 0,
                "experiences": experiences
            }
            
            if not os.path.exists(f"results/{MODEL_NAME}"):
                os.makedirs(f"results/{MODEL_NAME}")
            with open(f"results/{MODEL_NAME}/agentgym_{env_name}_{MODEL_NAME}.json", "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            print("\nCleaning up...")
            sandbox.shutdown()
            print("Done!")


if __name__ == "__main__":
    main()