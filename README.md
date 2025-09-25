# llmtalk

Tiny language-model demos for illustrating RNNs vs. transformers.

## Quick start

```bash
uv run python -m tinylm --help
```

## Train an experiment run

```bash
uv run python -m tinylm rnn --name rnn_demo --steps 2000 --eval-interval 200
```

- Creates `runs/rnn_demo/` with `config.json`, `model.pt`, `best_model.pt`, `best.json`, `vocab.json`, and `metrics.jsonl`.
- Override core hyperparameters with flags such as `--hidden-size`, `--seq-len`, `--batch-size`, or `--learning-rate`.
- Optional scheduler controls: `--lr-decay-steps` and `--lr-decay-gamma` apply a StepLR schedule.
- Re-run with `--overwrite` to replace an existing run directory.

## Generate text from a saved run

```bash
uv run python -m tinylm rnn prompt --name rnn_demo --prompt "Transformers will" --max-new-tokens 200 --temperature 0.8
uv run python -m tinylm rnn prompt --name rnn_demo --checkpoint best --prompt "Transformers will" --max-new-tokens 200 --temperature 0.8
```

Add `--seed` for deterministic sampling.

## Plot metrics (optional)

```bash
uv add seaborn matplotlib
uv run python -m tinylm rnn plot --name rnn_demo --metric val_ppl --output plots/rnn_demo_ppl.png
```

Plots are rendered with seaborn/matplotlib and can be saved to disk or displayed interactively.

## Transformer variant

Train a small transformer for comparison:

```bash
uv run python -m tinylm transformer --name tfm_demo --steps 8000 --eval-interval 500 --seq-len 256 --batch-size 64
```

Sample from the saved transformer run:

```bash
uv run python -m tinylm transformer prompt --name tfm_demo --prompt "Attention" --max-new-tokens 200 --temperature 0.8
```

## CLI shortcut

Install the console script locally for a shorter command:

```bash
uv run pip install -e .
tinylm rnn --name rnn_demo
tinylm transformer --name tfm_demo
```

The `tinylm` executable mirrors `python -m tinylm`.
