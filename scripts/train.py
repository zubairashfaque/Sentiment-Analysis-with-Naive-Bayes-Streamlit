#!/usr/bin/env python3
"""
Training script for sentiment analysis models.

Usage:
    python scripts/train.py --model naive_bayes --test-size 0.2
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sentiment_analysis.training.trainer import ModelTrainer
from sentiment_analysis.utils.logger import setup_logging
from sentiment_analysis.utils.metrics import calculate_metrics, print_metrics_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train sentiment analysis model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="naive_bayes",
        choices=["naive_bayes"],  # Will add more models later
        help="Model type to train"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/train.csv",
        help="Path to training data CSV file"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (0.0-1.0)"
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=None,
        help="Proportion of data for validation (optional)"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory to save trained model"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    print("🚀 Starting Sentiment Analysis Training")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Data:         {args.data}")
    print(f"Test Size:    {args.test_size}")
    print(f"Val Size:     {args.val_size if args.val_size else 'None'}")
    print(f"Random State: {args.random_state}")
    print(f"Output Dir:   {args.output_dir}")
    print("=" * 60)
    print()

    try:
        # Create trainer
        trainer = ModelTrainer(data_path=args.data)

        # Train model
        print("📚 Loading data and training model...")
        metrics = trainer.train(
            test_size=args.test_size,
            validation_size=args.val_size,
            random_state=args.random_state
        )

        # Print results
        print("\n✅ Training completed successfully!")
        print(f"\n📊 Results:")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"  Test Accuracy:  {metrics['test_accuracy']:.2%}")
        if metrics['val_accuracy']:
            print(f"  Val Accuracy:   {metrics['val_accuracy']:.2%}")

        print(f"\n📈 Per-Class Test Accuracy:")
        for sentiment, acc in metrics['test_per_class'].items():
            print(f"  {sentiment.capitalize():<10}: {acc:.2%}")

        print(f"\n⏱️  Training Time: {metrics['training_time_seconds']:.2f}s")

        # Save model
        print(f"\n💾 Saving model...")
        model_path = trainer.save_model(output_dir=args.output_dir)
        print(f"  Model saved to: {model_path}")

        print("\n" + "=" * 60)
        print("✨ Training pipeline completed successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ Error: Data file not found")
        print(f"  {e}")
        print(f"\n💡 Make sure the data file exists at: {args.data}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during training:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
