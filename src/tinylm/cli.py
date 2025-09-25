from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tinylm.data import CharVocab, prepare_char_data
from tinylm.models import MODEL_REGISTRY

RUNS_DIR = Path("runs")
COMMANDS = {"train", "prompt", "plot"}


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    if not argv or argv[0] in {"-h", "--help"}:
        print_general_help()
        return

    model_id = argv[0]
    remaining = argv[1:]

    if remaining and remaining[0] in COMMANDS:
        command = remaining[0]
        remainder_args = remaining[1:]
    else:
        command = "train"
        remainder_args = remaining

    handler = get_model_handler(model_id)

    if remainder_args and remainder_args[0] in {"-h", "--help"}:
        parser = build_parser_for_command(model_id, command)
        parser.print_help()
        return

    parser = build_parser_for_command(model_id, command)
    args = parser.parse_args(remainder_args)

    if command == "train":
        train_command(model_id, handler, args)
    elif command == "prompt":
        prompt_command(model_id, handler, args)
    elif command == "plot":
        plot_command(model_id, handler, args)
    else:  # pragma: no cover - should be unreachable
        raise SystemExit(f"Unknown command '{command}'")


def build_train_parser(model_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"tinylm {model_id} train", description="Train a model run and save artefacts"
    )
    parser.add_argument("--name", required=True, help="Name of the run directory under runs/")
    parser.add_argument("--steps", type=int, default=2000, help="Total optimisation steps")
    parser.add_argument("--eval-interval", type=int, default=200, help="Evaluation interval in steps")
    parser.add_argument("--embed-dim", type=int, default=None, help="Embedding dimension override")
    parser.add_argument("--hidden-size", type=int, default=None, help="Hidden size override")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length for BPTT")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="SGD learning rate")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of model layers")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads (transformer)")
    parser.add_argument("--ff-dim", type=int, default=None, help="Transformer feed-forward width")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--device", type=str, default=None, help="Device override (auto/cpu/cuda/mps)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay strength")
    parser.add_argument("--lr-decay-steps", type=int, default=None, help="StepLR decay interval")
    parser.add_argument("--lr-decay-gamma", type=float, default=None, help="StepLR decay factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing run directory with the same name"
    )
    return parser


def build_prompt_parser(model_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"tinylm {model_id} prompt", description="Generate text from a saved run"
    )
    parser.add_argument("--name", required=True, help="Run name (or path) to load")
    parser.add_argument("--prompt", default="", help="Prompt text to condition on")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Tokens to sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument(
        "--checkpoint",
        choices=["final", "best"],
        default="final",
        help="Which saved checkpoint to use",
    )
    return parser


def build_plot_parser(model_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"tinylm {model_id} plot", description="Plot training metrics using seaborn"
    )
    parser.add_argument("--name", required=True, help="Run name (or path) to visualise")
    parser.add_argument(
        "--metric",
        default="val_ppl",
        choices=["val_ppl", "mean_loss", "grad_norm", "interval_seconds"],
        help="Metric to plot over training steps",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path for saving the plot image")
    return parser


def print_general_help() -> None:
    models = ", ".join(sorted(MODEL_REGISTRY.keys()))
    commands = ", ".join(sorted(COMMANDS))
    message = (
        "Usage: tinylm <model> [command] [options]\n\n"
        "Models:\n  " + models + "\n\n"
        "Commands (default: train):\n  " + commands + "\n\n"
        "Examples:\n"
        "  tinylm rnn --name demo --steps 2000\n"
        "  tinylm transformer --name tfm_demo --steps 8000 --seq-len 256\n"
        "  tinylm rnn prompt --name demo --prompt 'Hello'\n"
        "  tinylm transformer plot --name tfm_demo --metric val_ppl\n"
    )
    print(message)


def build_parser_for_command(model_id: str, command: str) -> argparse.ArgumentParser:
    if command == "train":
        return build_train_parser(model_id)
    if command == "prompt":
        return build_prompt_parser(model_id)
    if command == "plot":
        return build_plot_parser(model_id)
    raise SystemExit(f"Unknown command '{command}'")


def train_command(model_id: str, handler, args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.name)
    ensure_run_dir(run_dir, overwrite=args.overwrite)

    print(f"Preparing data for run '{args.name}'...")
    data = prepare_char_data()
    overrides = {
        "embed_dim": args.embed_dim,
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "device": args.device,
        "weight_decay": args.weight_decay,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_decay_gamma": args.lr_decay_gamma,
        "seed": args.seed,
    }
    config = handler.build_config(vocab_size=data.vocab.size, **overrides)
    model = handler.create_model(config)

    rng = np.random.default_rng(args.seed)
    print(
        f"Training {model_id} for {args.steps} steps (eval every {args.eval_interval}) with "
        f"vocab={data.vocab.size}, batch={config.batch_size}, seq_len={config.seq_len}"
    )

    def log_eval(record: Dict[str, float]) -> None:
        step = int(record["step"])
        val_ppl = record["val_ppl"]
        mean_loss = record["mean_loss"]
        grad_norm = record["grad_norm"]
        interval_seconds = record["interval_seconds"]
        lr = record.get("lr")
        print(
            f"step={step:6d} | val_ppl={val_ppl:7.4f} | loss={mean_loss:6.4f} | "
            f"grad_norm={grad_norm:6.3f} | lr={lr:.4e} | +{interval_seconds:5.2f}s"
        )

    result = handler.train(
        model,
        data,
        steps=args.steps,
        eval_interval=args.eval_interval,
        rng=rng,
        on_eval=log_eval,
    )

    save_run(
        run_dir,
        model_id=model_id,
        model=model,
        config=handler.config_to_dict(config),
        vocab=data.vocab,
        result=result,
        handler=handler,
    )

    best_step = getattr(result, "best_step", None)
    best_metric = getattr(result, "best_metric", None)
    if best_step is not None and best_metric is not None:
        print(
            f"Best val_ppl={best_metric:.4f} at step {int(best_step)}; "
            f"checkpoint saved to {handler.best_state_path(run_dir).name}"
        )

    print(f"Run saved to {run_dir}")


def prompt_command(model_id: str, handler, args: argparse.Namespace) -> None:
    run_dir = locate_run_dir(args.name)
    run_data = load_run(run_dir)
    handler = get_model_handler(run_data["model_id"])

    config = handler.config_from_dict(run_data["model_config"])
    vocab = CharVocab.from_serializable(run_data["vocab"])
    model = handler.create_model(config)
    checkpoint_path = (
        handler.state_path(run_dir)
        if args.checkpoint == "final"
        else handler.best_state_path(run_dir)
    )
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint '{args.checkpoint}' not found at {checkpoint_path}")
    handler.load_state(model, checkpoint_path)

    generated = handler.sample(
        model,
        vocab,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(generated)


def plot_command(model_id: str, handler, args: argparse.Namespace) -> None:
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Seaborn/matplotlib are required for plotting. Install with `uv add seaborn matplotlib`."
        ) from exc

    run_dir = locate_run_dir(args.name)
    metrics = load_metrics(run_dir)
    if not metrics:
        raise SystemExit("No metrics found for the requested run.")

    steps = [entry["step"] for entry in metrics]
    values = [entry[args.metric] for entry in metrics]

    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(x=steps, y=values)
    ax.set_xlabel("Steps")
    ax.set_ylabel(args.metric)
    ax.set_title(f"{run_dir.name}: {args.metric}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)
        print(f"Plot written to {args.output}")
    else:
        plt.show()


def get_model_handler(model_id: str):
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:  # pragma: no cover
        raise SystemExit(f"Unknown model '{model_id}'. Available: {sorted(MODEL_REGISTRY.keys())}") from exc


def resolve_run_dir(run_name: str) -> Path:
    path = Path(run_name)
    if path.is_absolute():
        return path
    return RUNS_DIR / run_name


def ensure_run_dir(run_dir: Path, *, overwrite: bool) -> None:
    if run_dir.exists():
        if overwrite:
            import shutil

            shutil.rmtree(run_dir)
        else:
            raise SystemExit(f"Run directory {run_dir} already exists. Use --overwrite to replace.")
    run_dir.mkdir(parents=True, exist_ok=True)


def save_run(
    run_dir: Path,
    *,
    model_id: str,
    model: Any,
    config: Dict[str, Any],
    vocab: CharVocab,
    result: Any,
    handler: Any,
) -> None:
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "model_config": config,
            },
            f,
            indent=2,
        )

    with open(run_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab.to_serializable(), f, indent=2)

    metrics: List[Dict[str, float]] = result.metrics

    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as f:
        for entry in metrics:
            f.write(json.dumps(entry) + "\n")

    handler.save_state(model, handler.state_path(run_dir))

    best_state = getattr(result, "best_state", None)
    best_step = getattr(result, "best_step", None)
    best_metric = getattr(result, "best_metric", None)
    if best_state is not None:
        handler.save_state_dict(best_state, handler.best_state_path(run_dir))
        with open(run_dir / "best.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": best_step,
                    "metric": best_metric,
                },
                f,
                indent=2,
            )


def load_run(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.json"
    vocab_path = run_dir / "vocab.json"
    if not config_path.exists() or not vocab_path.exists():
        raise SystemExit(f"Run directory {run_dir} is missing config or vocab files")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return {
        "model_id": config["model_id"],
        "model_config": config["model_config"],
        "vocab": vocab,
    }


def load_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    with open(metrics_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def locate_run_dir(name_or_path: str) -> Path:
    explicit = Path(name_or_path)
    if explicit.exists():
        return explicit
    candidate = RUNS_DIR / name_or_path
    if candidate.exists():
        return candidate
    raise SystemExit(f"Cannot find run directory for '{name_or_path}'")


if __name__ == "__main__":  # pragma: no cover
    main()
