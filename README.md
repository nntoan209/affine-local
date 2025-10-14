# Affine

## Affine evaluation

Deploy your model with sglang.

```bash
# sglang deploy
```

Change the model name, number of testcases, and sglang port in [affine_evaluator.py](affine/quixand/examples/affine_evaluator.py).

Run the evaluator. You can specify the environments to evaluate.

```bash
python affine/quixand/examples/affine_evaluator.py -envs sat ded abd
```


## AgentGym evaluation

Deploy your model with sglang.

```bash
# sglang deploy
```

Change the model name and sglang port in [agentgym_evaluator.py](affine/quixand/examples/agentgym_evaluator.py).

Run the evaluator. You can specify the environments to evaluate.

```bash
python affine/quixand/examples/affine_evaluator.py -envs alfworld webshop babyai sciworld textcraft
```
