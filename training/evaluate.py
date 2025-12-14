import argparse
import importlib.util
from pathlib import Path
from typing import Any


def _load_alfa_eval_module() -> Any:
    base_dir = Path(__file__).resolve().parents[2]
    eval_path = base_dir / "evaluate.py"
    if not eval_path.exists():
        raise FileNotFoundError(str(eval_path))
    spec = importlib.util.spec_from_file_location(
        "evaluate", str(eval_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_evaluation(
    model_path: str,
    data_dir: str,
    output_dir: str,
    mask_length: int,
    precision_mode: str,
    precision_quantile: float,
    recalibrate: bool,
    recalibrate_method: str,
    strict_level: str,
    auto_adjust_threshold: bool,
) -> str:
    module = _load_alfa_eval_module()
    evaluator = module.MGGATEvaluatorStrict(
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        mask_length=mask_length,
        precision_mode=precision_mode,
        precision_quantile=precision_quantile,
    )
    evaluator.run_evaluation(
        recalibrate=recalibrate,
        recalibrate_method=recalibrate_method,
        strict_level=strict_level,
        auto_adjust_threshold=auto_adjust_threshold,
    )
    return str(evaluator.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict MG-GAT-BiLSTM evaluation entry for arbitrary datasets"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (best_model.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing test_sequences_flight_*.npy and labels",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Images",
        help="Base directory for saving evaluation outputs",
    )
    parser.add_argument(
        "--mask_length",
        type=int,
        default=150,
        help="Temporal mask length used during evaluation",
    )
    parser.add_argument(
        "--precision_mode",
        type=str,
        default="normal",
        choices=["normal", "high_precision"],
        help="Precision/recall trade-off mode",
    )
    parser.add_argument(
        "--precision_quantile",
        type=float,
        default=0.995,
        help="Quantile for precision-oriented threshold adjustment",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Recalibrate threshold on validation scores without using labels",
    )
    parser.add_argument(
        "--recalibrate_method",
        type=str,
        default="pot",
        choices=["pot", "mad", "iqr", "zscore", "quantile"],
        help="Unsupervised thresholding method",
    )
    parser.add_argument(
        "--strict_level",
        type=str,
        default="normal",
        choices=["normal", "strict", "very_strict"],
        help="Threshold strictness level",
    )
    parser.add_argument(
        "--auto_adjust_threshold",
        action="store_true",
        help="Automatically adjust threshold based on score distribution",
    )
    args = parser.parse_args()
    out_dir = run_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mask_length=args.mask_length,
        precision_mode=args.precision_mode,
        precision_quantile=args.precision_quantile,
        recalibrate=args.recalibrate,
        recalibrate_method=args.recalibrate_method,
        strict_level=args.strict_level,
        auto_adjust_threshold=args.auto_adjust_threshold,
    )
    print("evaluation_output_dir:", out_dir)


if __name__ == "__main__":
    main()

