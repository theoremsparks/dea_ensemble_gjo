"""
Compare all trained ensemble models and create comprehensive comparison reports.
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    def __init__(self, outputs_dir: str = None):
        """
        Initialize model comparator

        Args:
            outputs_dir: Directory containing model outputs. If None, uses project root.
        """
        # Set outputs directory - go up one level from src to project root
        if outputs_dir is None:
            # Get the project root (go up from src/)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outputs_dir = os.path.join(project_root, "outputs")

        self.outputs_dir = outputs_dir

        # Model names and colors for visualization
        self.models = {
            "xgb": "XGBoost",
            "gbr": "Gradient Boosting",
            "lgbm": "LightGBM",
            "rf": "Random Forest",
            "et": "Extra Trees",
            "ada": "AdaBoost"
        }

        self.colors = {
            "xgb": "#1f77b4",  # Blue
            "gbr": "#ff7f0e",  # Orange
            "lgbm": "#2ca02c",  # Green
            "rf": "#d62728",  # Red
            "et": "#9467bd",  # Purple
            "ada": "#8c564b"  # Brown
        }

        self.model_dirs = {}
        self.metrics_data = {}

        logger.info(f"ModelComparator initialized")
        logger.info(f"Outputs directory: {self.outputs_dir}")
        logger.info(f"Directory exists: {os.path.exists(self.outputs_dir)}")

    def find_latest_model_dirs(self):
        """Find the latest output directories for each model"""
        logger.info("Finding latest model outputs...")
        logger.info(f"Searching in: {self.outputs_dir}")

        # Check if outputs directory exists
        if not os.path.exists(self.outputs_dir):
            logger.error(f"Outputs directory does not exist: {self.outputs_dir}")
            logger.error("Please run models first using: python src/run_pipeline.py")
            return self.model_dirs

        # List all items in outputs for debugging
        try:
            all_items = os.listdir(self.outputs_dir)
            logger.info(f"Found {len(all_items)} items in outputs folder:")
            for item in all_items:
                item_path = os.path.join(self.outputs_dir, item)
                item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                logger.info(f"  [{item_type}] {item}")
        except Exception as e:
            logger.error(f"Error listing outputs directory: {e}")

        for model_code, model_name in self.models.items():
            # Try multiple search patterns
            patterns = [
                # Flat structure: outputs/xgb_gjo_*
                os.path.join(self.outputs_dir, f"{model_code}_gjo_*"),
                # Nested structure: outputs/xgb/xgb_gjo_*
                os.path.join(self.outputs_dir, model_code, f"{model_code}_gjo_*"),
                # Any directory with model code
                os.path.join(self.outputs_dir, f"*{model_code}*")
            ]

            found_dirs = []
            for pattern in patterns:
                try:
                    dirs = glob.glob(pattern)
                    for dir_path in dirs:
                        if os.path.isdir(dir_path) and dir_path not in found_dirs:
                            found_dirs.append(dir_path)
                except:
                    continue

            if found_dirs:
                # Get the most recent directory
                latest_dir = max(found_dirs, key=os.path.getmtime)
                self.model_dirs[model_code] = latest_dir
                folder_name = os.path.basename(latest_dir)
                logger.info(f"  ✓ {model_name}: {folder_name}")
            else:
                logger.warning(f"  ✗ {model_name}: Not found")

        logger.info(f"\nTotal model directories found: {len(self.model_dirs)}/6")
        return self.model_dirs

    def load_model_metrics(self):
        """Load metrics from all model output files"""
        logger.info("Loading model metrics...")

        for model_code, model_dir in self.model_dirs.items():
            # Find metrics file
            metrics_files = glob.glob(os.path.join(model_dir, f"{model_code}_metrics_*.txt"))

            if metrics_files:
                metrics_file = max(metrics_files, key=os.path.getmtime)  # Get latest
                metrics = self._parse_metrics_file(metrics_file)
                self.metrics_data[model_code] = metrics
                logger.info(f"  ✓ {self.models[model_code]}: Loaded metrics")
            else:
                logger.warning(f"  ✗ {self.models[model_code]}: No metrics file found")

        return self.metrics_data

    def _parse_metrics_file(self, filepath: str) -> Dict:
        """Parse metrics from text file"""
        metrics = {"train": {}, "test": {}}
        current_section = None

        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()

                # Detect section headers
                if "Training Data Metrics" in line:
                    current_section = "train"
                    continue
                elif "Testing Data Metrics" in line:
                    current_section = "test"
                    continue
                elif "====" in line or "====" in line:
                    continue

                # Parse metrics lines
                if current_section and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        metric_name = parts[0].strip().replace(" ", "_").lower()
                        metric_value = parts[1].strip()

                        # Remove % sign and convert to float
                        if "%" in metric_value:
                            metric_value = metric_value.replace("%", "")

                        try:
                            metrics[current_section][metric_name] = float(metric_value)
                        except ValueError:
                            metrics[current_section][metric_value] = metric_value

        except Exception as e:
            logger.error(f"Error parsing metrics file {filepath}: {e}")

        return metrics

    def create_comparison_table(self):
        """Create comparison table of all models"""
        if not self.metrics_data:
            logger.error("No metrics data loaded")
            return None

        # Prepare data for DataFrame
        rows = []
        for model_code, metrics in self.metrics_data.items():
            if "test" in metrics:
                row = {
                    "Model": self.models[model_code],
                    "R2": metrics["test"].get("r2", np.nan),
                    "RMSE": metrics["test"].get("rmse", np.nan),
                    "MAE": metrics["test"].get("mae", np.nan),
                    "MAPE": metrics["test"].get("mape", np.nan),
                    "sMAPE": metrics["test"].get("smape", np.nan),
                    "R": metrics["test"].get("r", np.nan)
                }
                rows.append(row)

        if not rows:
            logger.error("No test metrics found in any model")
            return None, None

        df = pd.DataFrame(rows)

        # Sort by R2 (descending)
        df = df.sort_values("R2", ascending=False).reset_index(drop=True)

        # Determine best model
        best_model = df.iloc[0]["Model"]
        best_r2 = df.iloc[0]["R2"]

        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON TABLE")
        logger.info("=" * 80)
        logger.info(df.to_string(index=False))
        logger.info(f"\n🏆 Best Model: {best_model} (R2 = {best_r2:.4f})")

        return df, best_model


    def generate_comprehensive_report(self, df_comparison, best_model):
        """Generate comprehensive comparison report"""
        if df_comparison is None or best_model is None:
            logger.error("Cannot generate report without comparison data")
            return

        # Create in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_dir = os.path.join(project_root, "comparison_results")
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = os.path.join(report_dir, f"model_comparison_report_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEA-ENSEMBLE MODEL COMPARISON REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Models Compared: {len(self.metrics_data)}\n")
            f.write(f"Best Model: {best_model}\n")
            f.write(f"Best R2: {df_comparison.iloc[0]['R2']:.4f}\n\n")

            # Detailed Comparison
            f.write("DETAILED COMPARISON\n")
            f.write("-" * 40 + "\n")
            f.write(df_comparison.to_string(index=False) + "\n\n")

            # Model Directories
            f.write("MODEL OUTPUT DIRECTORIES\n")
            f.write("-" * 40 + "\n")
            for model_code, model_dir in self.model_dirs.items():
                f.write(f"{self.models[model_code]:<20}: {model_dir}\n")

            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write(f"1. Primary Model: {best_model}\n")
            f.write("   - Highest R2 score\n")
            f.write("   - Best overall performance\n\n")

            # Find runner-up
            if len(df_comparison) > 1:
                runner_up = df_comparison.iloc[1]["Model"]
                f.write(f"2. Alternative Model: {runner_up}\n")
                f.write("   - Good performance with different characteristics\n")
                f.write("   - Useful for ensemble or validation\n\n")

            f.write("3. Considerations:\n")
            f.write("   - Check computational requirements\n")
            f.write("   - Consider model interpretability\n")
            f.write("   - Evaluate training time vs performance trade-off\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Comprehensive report saved to: {report_path}")

        # Also save as CSV
        csv_path = os.path.join(report_dir, f"model_comparison_{timestamp}.csv")
        df_comparison.to_csv(csv_path, index=False)
        logger.info(f"Comparison data saved to: {csv_path}")

    def run_complete_comparison(self):
        """Run complete comparison analysis"""
        logger.info("=" * 80)
        logger.info("MODEL COMPARISON ANALYSIS")
        logger.info("=" * 80)

        # Step 1: Find model directories
        self.find_latest_model_dirs()

        if not self.model_dirs:
            logger.error("No model directories found!")
            logger.error("Please run models first using: python src/run_pipeline.py")
            return

        # Step 2: Load metrics
        self.load_model_metrics()

        if not self.metrics_data:
            logger.error("No metrics data loaded!")
            return

        # Step 3: Create comparison table
        df_comparison, best_model = self.create_comparison_table()

        if df_comparison is not None and best_model is not None:

            # Step 4: Generate comprehensive report
            self.generate_comprehensive_report(df_comparison, best_model)

            logger.info("\n✅ Model comparison completed successfully!")
            logger.info(f"Best model: {best_model}")
            logger.info(f"Reports saved to: comparison_results/")
        else:
            logger.error("Failed to create comparison table")


def main():
    """Main entry point for model comparison"""
    import argparse

    parser = argparse.ArgumentParser(description="Compare all trained ensemble models")
    parser.add_argument("--outputs-dir", default=None,
                        help="Directory containing model outputs (default: project_root/outputs)")
    parser.add_argument("--report-dir", default=None,
                        help="Directory for comparison reports (default: project_root/comparison_results)")

    args = parser.parse_args()

    # Run comparison
    comparator = ModelComparator(outputs_dir=args.outputs_dir)
    comparator.run_complete_comparison()


if __name__ == "__main__":
    main()