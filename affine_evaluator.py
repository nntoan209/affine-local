import asyncio
import argparse
import aiohttp
import json
import random
from loguru import logger
from dataclasses import dataclass, asdict
from typing import List
import os

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
                       default="http://localhost:10000/v1",
                       help='Model service URL (required with --model)')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of samples to evaluate')
    parser.add_argument('--concurrency', type=int, default=8,
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
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
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

    def save_results(self):
        """Save current results to output file."""
        try:
            with open(self.output_file, 'w') as f:
                for result in self.results:
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
        perfect_scores = sum(1 for r in self.results if r.score == 1.0)
        return perfect_scores / len(self.results)
        
    async def process_sample(self, index: int):
        """
        Process a single sample with retries until score is 1 or max retries reached.
        
        Args:
            index: Dataset index to process
            
        Returns:
            SampleResult if successful (score=1), None if discarded
        """
        async with self.semaphore:
            try:
                # Generate challenge
                challenge = await self.task.generate(index)
                    
                # Call LLM API
                llm_response = await call_llm_api(self.model, self.base_url, challenge.prompt)
                
                # Evaluate the response
                evaluation = await self.task.evaluate(llm_response, challenge)
                
                # If perfect score, return success
                result = SampleResult(
                    index=index,
                    prompt=challenge.prompt,
                    response=llm_response,
                    score=evaluation,
                )
                logger.info(f"Sample {index} completed")
                return result
                
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
                    # Successful result with score=1
                    self.results.append(result)
                    self.completed_indices.add(result.index)
                    self.samples_since_save += 1

                    # Print running accuracy
                    accuracy = self.get_running_accuracy()
                    logger.info(
                        f"Running accuracy: {accuracy:.2%} "
                        f"({sum(1 for r in self.results if r.score == 1.0)}/{len(self.results)})"
                    )

                    # Periodic save
                    if self.samples_since_save >= self.save_interval:
                        self.save_results()
                        self.samples_since_save = 0
            
            # Add new tasks to maintain concurrency
            while (self.current_index < self.dataset_size and 
                   len(tasks) < self.max_concurrent):
                new_task = asyncio.create_task(self.process_sample(self.current_index))
                tasks.add(new_task)
                self.current_index += 1

        # Final save
        self.save_results()
        self.save_summary()
        
        return self.results

async def main():
    """Main evaluation function"""
    args = parse_args()

    from affine.envs import SAT, DED, ABD
    from datasets import load_dataset

    ds = load_dataset("satpalsr/rl-python", split="train")
    ds = ds.select(random.sample(range(len(ds)), args.num_samples))

    # Map environment names to actual classes
    ENVIRONMENTS = {
        'sat': SAT,
        'ded': DED,
        'abd': ABD,
    }

    for env in args.envs:
        env_class = ENVIRONMENTS[env]
        if env in ["abd", "ded"]:
            task = env_class(ds)
        else:
            task = env_class()

        evaluator = ConcurrentEvaluator(
            model=args.model,
            base_url=args.base_url,
            task=task,
            dataset_size=len(ds),
            max_concurrent=args.concurrency,
            save_interval=16,
            output_file=f"results_{env}.jsonl",
        )

        results = await evaluator.run()


if __name__ == "__main__":
    asyncio.run(main())