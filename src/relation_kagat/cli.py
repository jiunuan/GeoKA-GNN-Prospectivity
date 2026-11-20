import argparse
from pathlib import Path
from .pipeline import load_config, prepare, cross_validate, train, predict


def _print_threshold(threshold):
    print(f"Classification threshold = {threshold} (Youden J on pooled three-fold OOF scores)")


def main():
    parser = argparse.ArgumentParser(description="Mineral prospectivity mapping with GeoKA-GNN")
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "prepare": "Prepare inputs and fix the three development folds and outer test set.",
        "cross-validate": "Run three development folds and select the raw-score cutoff by pooled OOF Youden J.",
        "train": "Refit on all development data and evaluate the outer test at the cutoff selected by cross-validation.",
        "predict": "Export a continuous area-percentile map after the raster quality checks.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--config", required=True, type=Path)
    demo = commands.add_parser("demo", help="Generate synthetic inputs and smoke-test training; no prediction claim")
    demo.add_argument("--work", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--template", type=Path, default=Path("configs/default.yaml"))
    demo.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.command == "demo":
        from .synthetic import make_demo
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        path = make_demo(args.work, args.template, args.epochs)
        cfg = load_config(path)
        prepare(cfg)
        _print_threshold(train(cfg)["threshold"])
        print(f"Synthetic training smoke test finished. Config: {path}")
        print("No prospectivity experiment or raster release is claimed by this demo.")
    else:
        cfg = load_config(args.config)
        if args.command == "cross-validate":
            _print_threshold(cross_validate(cfg)["threshold_selection"]["threshold"])
        elif args.command == "train":
            _print_threshold(train(cfg)["threshold"])
        else:
            {"prepare": prepare, "predict": predict}[args.command](cfg)


if __name__ == "__main__":
    main()
