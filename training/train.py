import argparse
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional


def _load_alfa_train_module() -> Any:
    base_dir = Path(__file__).resolve().parents[2]
    train_path = base_dir / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(str(train_path))
    spec = importlib.util.spec_from_file_location("train", str(train_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_config(
    base_config: Dict[str, Any],
    data_dir: Optional[str],
    output_dir: Optional[str],
    batch_size: Optional[int],
    epochs: Optional[int],
    learning_rate: Optional[float],
) -> Dict[str, Any]:
    config = dict(base_config)
    if data_dir is not None:
        config["data_dir"] = data_dir
    if output_dir is not None:
        config["output_dir"] = output_dir
    if batch_size is not None:
        config["batch_size"] = batch_size
    if epochs is not None:
        config["max_epochs"] = epochs
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    return config


def run_training(
    data_dir: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> Dict[str, Any]:
    module = _load_alfa_train_module()
    base_config = module.get_default_config()
    if config_path is not None:
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            base_config.update(user_cfg)
    config = _build_config(base_config, data_dir, output_dir, batch_size, epochs, learning_rate)
    module.set_seed(2024)
    trainer = module.MGGATTrainer(config)
    trainer.load_data()
    trainer.build_model()
    best_loss = trainer.train()
    threshold = trainer.evaluate_and_calibrate()
    result = {
        "best_val_loss": float(best_loss),
        "threshold": float(threshold),
        "model_dir": str(trainer.output_dir),
        "data_dir": str(config["data_dir"]),
    }
    summary_path = Path(trainer.output_dir) / "training_summary_generic.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic MG-GAT-BiLSTM training entry for arbitrary datasets"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed train/val numpy arrays",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file overriding default hyperparameters",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override maximum training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base directory for saving trained models and logs",
    )
    args = parser.parse_args()
    result = run_training(
        data_dir=args.data_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print("best_val_loss:", result["best_val_loss"])
    print("threshold:", result["threshold"])
    print("model_dir:", result["model_dir"])


if __name__ == "__main__":
    main()

