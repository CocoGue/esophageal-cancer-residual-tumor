"""
Command-line interface for ct_residual_disease.

Provides commands for:
- training a logistic regression model
- running inference on new data
"""

import argparse
import pickle
import sys

from ct_residual_disease.pipelines.features_pipeline import (
    build_features_and_labels,
)
from ct_residual_disease.pipelines.training_pipeline import (
    train_logistic_pipeline,
)
from ct_residual_disease.pipelines.inference_pipeline import (
    run_inference_pipeline,
)


# ---------------------------------------------------------------------
# Training command
# ---------------------------------------------------------------------
def _run_train(args: argparse.Namespace) -> None:
    """
    Run the training pipeline from CLI arguments.
    """
    X, y = build_features_and_labels(
        csv_path=args.csv_path,
        feature_mode=args.feature_mode,
        encoding="logistic",
        drop_first=args.drop_first,
    )

    results = train_logistic_pipeline(
        X=X,
        y=y,
        feature_mode=args.feature_mode,
        scale_numerical=args.scale_numerical,
        add_intercept=args.add_intercept,
        add_elasticnet_regularization=args.elasticnet,
        reglog_regularization=args.regularization,
    )

    # Save artifacts
    with open(args.model_path, "wb") as f:
        pickle.dump(results["model"], f)

    with open(args.scaler_path, "wb") as f:
        pickle.dump(results["scaler"], f)

    with open(args.threshold_path, "w") as f:
        f.write(str(results["threshold"]))

    print("Training completed successfully.")
    print(f"Model saved to: {args.model_path}")
    print(f"Scaler saved to: {args.scaler_path}")
    print(f"Threshold saved to: {args.threshold_path}")


# ---------------------------------------------------------------------
# Inference command
# ---------------------------------------------------------------------
def _run_infer(args: argparse.Namespace) -> None:
    """
    Run the inference pipeline from CLI arguments.
    """
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(args.threshold_path, "r") as f:
        threshold = float(f.read())

    prob_metrics, cls_metrics = run_inference_pipeline(
        csv_path=args.csv_path,
        model=model,
        threshold=threshold,
        feature_mode=args.feature_mode,
        encoding="logistic",
        drop_first=args.drop_first,
        scaler=scaler,
    )

    print("Inference completed.")
    print("\nProbability metrics:")
    for k, v in prob_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nClassification metrics:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ct_residual_disease CLI"
    )

    subparsers = parser.add_subparsers(dest="command")

    # ------------------ TRAIN ------------------
    train_parser = subparsers.add_parser(
        "train", help="Train a logistic regression model"
    )
    train_parser.add_argument("csv_path", type=str)
    train_parser.add_argument("--feature-mode", default="combined")
    train_parser.add_argument("--scale-numerical", action="store_true")
    train_parser.add_argument("--drop-first", action="store_true")
    train_parser.add_argument("--add-intercept", action="store_true")
    train_parser.add_argument("--elasticnet", action="store_true")
    train_parser.add_argument("--regularization", type=float, default=1.0)

    train_parser.add_argument("--model-path", default="trained_model/model.pkl")
    train_parser.add_argument("--scaler-path", default="trained_model/scaler.pkl")
    train_parser.add_argument("--threshold-path", default="trained_model/threshold.txt")

    train_parser.set_defaults(func=_run_train)

    # ------------------ INFER ------------------
    infer_parser = subparsers.add_parser(
        "infer", help="Run inference on new data"
    )
    infer_parser.add_argument("csv_path", type=str)
    infer_parser.add_argument("--feature-mode", default="combined")
    infer_parser.add_argument("--drop-first", action="store_true")

    infer_parser.add_argument("--model-path", default="trained_model/model.pkl")
    infer_parser.add_argument("--scaler-path", default="trained_model/scaler.pkl")
    infer_parser.add_argument("--threshold-path", default="trained_model/threshold.txt")

    infer_parser.set_defaults(func=_run_infer)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
