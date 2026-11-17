import argparse
from pathlib import Path
from .pipeline import load_config, prepare, cross_validate, train, predict


def main():
    parser = argparse.ArgumentParser(description="Mineral prospectivity mapping with GeoKA-GNN")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "cross-validate", "train", "predict"]:
        command = commands.add_parser(name)
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
        train(cfg)
        print(f"Synthetic training smoke test finished. Config: {path}")
        print("No prospectivity experiment or raster release is claimed by this demo.")
    else:
        cfg = load_config(args.config)
        {"prepare": prepare, "cross-validate": cross_validate, "train": train, "predict": predict}[args.command](cfg)


if __name__ == "__main__":
    main()
