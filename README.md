# Affine

## AgentGym evaluation

```bash
python agentgym_evaluator.py -envs webshop alfworld babyai sciworld textcraft --model your_model_name --base-url http://localhost:{{sglang_port}}/v1 --concurrency 32
```

## Affine evaluation
```bash
python affine_evaluator.py -envs sat abd ded --model your_model_name --base-url http://localhost:{{sglang_port}}/v1 --num-samples 500 --concurrency 32
```