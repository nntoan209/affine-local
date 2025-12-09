import asyncio
import argparse
import aiohttp
import json
import random
from loguru import logger
from dataclasses import dataclass, asdict
from typing import List
import os
from datasets import load_dataset

RETRIES = 1

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
                       default="https://llm.chutes.ai/v1",
                       help='Model service URL (required with --model)')
    parser.add_argument('--concurrency', type=int, default=32,
                       help='Number of concurrent evaluations')
    args = parser.parse_args()

    # Validation
    if args.model and not args.base_url:
        parser.error('--base-url is required when using --model')

    return args

async def call_llm_api(model: str, base_url: str, prompt: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.post(
        f"{base_url}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('CHUTES_API_KEY')}",
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "stream": False
        }
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to call LLM API: {response.status}")
            res = await response.json()
            return res["choices"][0]["message"]["content"]

@dataclass
class SampleResult:
    """Structure to hold results for each sample."""
    index: int
    prompt: str
    response: str
    score: float

class ConcurrentEvaluator:
    """Manages concurrent evaluation of dataset samples."""
    
    def __init__(
        self,
        model: str,
        base_url: str,
        task,
        dataset_size: int,
        max_concurrent: int = 10,
        save_interval: int = 10,
        output_file: str = "results.jsonl",
    ):
        self.model = model
        self.base_url = base_url
        self.task = task
        self.dataset_size = dataset_size
        self.max_concurrent = max_concurrent
        self.save_interval = save_interval
        self.output_file = output_file
        
        # Track results and state
        self.results: List[SampleResult] = []
        self.completed_indices = set()
        self.current_index = 0
        self.samples_since_save = 0
        
        # Semaphore to limit concurrent tasks
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # get existing indices from output file
        if not os.path.exists(output_file):
            self.completed_indices = set()
        else:
            self.completed_indices = set(load_dataset("json", data_files=output_file, split="train").filter(lambda x: x["score"] > 0, num_proc=8)["index"])

    def save_results(self, result: SampleResult):
        """Save current results to output file."""
        try:
            with open(self.output_file, 'a') as f:
                if result.score == 1.0:
                    f.write(json.dumps(asdict(result)) + '\n')
            logger.info(f"Saved {len(self.results)} results to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def save_summary(self):
        """Save summary statistics to a separate file."""
        summary_file = self.output_file.replace('.jsonl', '_summary.json')
        
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        # Calculate statistics
        total_samples = len(self.results)
        scores = [r.score for r in self.results]
        avg_score = sum(scores) / total_samples if total_samples > 0 else 0
        perfect_scores = sum(1 for s in scores if s == 1.0)
        accuracy = perfect_scores / total_samples if total_samples > 0 else 0
        
        summary = {
            "total_samples": total_samples,
            "dataset_size": self.dataset_size,
            "average_score": avg_score,
            "perfect_scores": perfect_scores,
            "accuracy": accuracy,
            "score_distribution": {
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "avg": avg_score
            }
        }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary saved to {summary_file}")
            logger.info(f"Accuracy: {accuracy:.2%} ({perfect_scores}/{total_samples})")
        except Exception as e:
            logger.error(f"Error saving summary: {e}")

    def get_running_accuracy(self) -> float:
        """Calculate current accuracy from results."""
        if not self.results:
            return 0.0
        
        return sum([r.score for r in self.results]) / len(self.results)
        
    async def process_sample(self, index: int):
        """
        Process a single sample with retries until score is > 0 or max retries reached.
        
        Args:
            index: Dataset index to process
            
        Returns:
            SampleResult if successful (score>0), None if discarded
        """
        async with self.semaphore:
            try:
                # Generate challenge
                challenge = await self.task.generate(index)
                
                for _ in range(RETRIES):
                    # Call LLM API
                    llm_response = await call_llm_api(self.model, self.base_url, challenge.prompt)
                    
                    # Evaluate the response
                    score, test_result = await self.task.evaluate(llm_response, challenge)
                    
                    # If score > 0, return success
                    if score > 0:
                        result = SampleResult(
                            index=index,
                            prompt=challenge.prompt,
                            response=llm_response,
                            score=score,
                        )
                        logger.info(f"Sample {index} completed with score {score}")
                        return result
                    
                # If no score > 0, return failed sample
                logger.warning(f"Sample {index} failed after {RETRIES} retries")
                return SampleResult(
                    index=index,
                    prompt="",
                    response="",
                    score=0.0,
                )
                
            except Exception as e:
                logger.error(f"Error processing sample {index}: {e}")
                return None
    
    async def run(self):
        """
        Main execution loop: manages concurrent processing with dynamic task scheduling.
        """
        logger.info(
            f"Starting evaluation with {self.max_concurrent} concurrent workers"
        )
        
        # Initialize task queue
        tasks = set()
        
        # Fill initial task pool
        while self.current_index < self.dataset_size and len(tasks) < self.max_concurrent:
            if self.current_index not in self.completed_indices:
                task = asyncio.create_task(self.process_sample(self.current_index))
                tasks.add(task)
            self.current_index += 1
        
        # Process tasks dynamically
        while tasks:
            # Wait for at least one task to complete
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Process completed tasks
            for task in done:
                result = await task
                
                if result is not None:
                    # Successful result with score>0
                    self.results.append(result)

                    # save result
                    self.save_results(result)

                    # Print running accuracy
                    accuracy = self.get_running_accuracy()
                    logger.info(
                        f"Running accuracy: {accuracy:.2%}"
                    )
            
            # Add new tasks to maintain concurrency
            while (self.current_index < self.dataset_size and len(tasks) < self.max_concurrent):
                if self.current_index not in self.completed_indices:
                    new_task = asyncio.create_task(self.process_sample(self.current_index))
                    tasks.add(new_task)
                self.current_index += 1

        # Final save summary
        self.save_summary()
        
        return self.results

async def main():
    """Main evaluation function"""
    args = parse_args()

    from affine.envs.pi import CodeTask

    # Map environment names to actual classes
    ENVIRONMENTS = {
        'code': CodeTask,
    }

    model_name = args.model.split("/")[-1]

    for env in args.envs:
        env_class = ENVIRONMENTS[env]
        task = env_class()
        logger.info(f"Length of dataset: {len(task.dataset)}")

        evaluator = ConcurrentEvaluator(
            model=args.model,
            base_url=args.base_url,
            task=task,
            dataset_size=len(task.dataset),
            max_concurrent=args.concurrency,
            save_interval=16,
            output_file=f"results_{env}_{model_name}.jsonl",
        )

        results = await evaluator.run()


if __name__ == "__main__":
    asyncio.run(main())
