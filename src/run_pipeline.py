import sys
import platform
import os
import subprocess


def check_environment():
    print("=" * 60)
    print("PYCHARM ENVIRONMENT CHECK")
    print("=" * 60)

    # 1. Python Information
    print("\n1. PYTHON INFORMATION:")
    print(f"   Executable: {sys.executable}")
    print(f"   Version: {platform.python_version()}")
    print(f"   Full Version: {sys.version}")
    print(f"   Platform: {platform.platform()}")

    # 2. Virtual Environment
    print("\n2. VIRTUAL ENVIRONMENT:")
    venv_path = os.environ.get('VIRTUAL_ENV', None)
    if venv_path:
        print(f"   ✅ Using virtual environment: {venv_path}")
        print(f"   Virtual env name: {os.path.basename(venv_path)}")
    else:
        print("   ❌ Not using virtual environment (using system Python)")

    # 3. Project Paths
    print("\n3. PROJECT PATHS:")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Python path: {sys.path[:3]}...")  # First 3 entries

    # 4. Environment Variables (ML-specific)
    print("\n4. ENVIRONMENT VARIABLES:")
    env_vars = ['VIRTUAL_ENV', 'PYTHONPATH', 'PATH']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH' and len(value) > 100:
            value = value[:100] + "..."
        print(f"   {var}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    check_environment()


"""
Main pipeline script for DEA-Ensemble-GJO model.
Runs DEA analysis first, then ALL 6 ensemble models.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dea_model import run_dea_pipeline
from xgb_model import run_xgb_pipeline
from gbr_model import run_gbr_pipeline
from lgb_model import run_lgbm_pipeline
from rf_model import run_rf_pipeline
from et_model import run_et_pipeline
from ada_model import run_ada_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DEAEnsemblePipeline:
    def __init__(self, config: Optional[Dict] = None):

        self.config = config or self.get_default_config()

        # Get project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Setup paths
        self.data_raw_dir = os.path.join(self.project_root, "data", "raw")
        self.data_processed_dir = os.path.join(self.project_root, "data", "processed")
        self.outputs_dir = os.path.join(self.project_root, "outputs")

        # Create directories if they don't exist
        os.makedirs(self.data_raw_dir, exist_ok=True)
        os.makedirs(self.data_processed_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        logger.info(f"DEA-Ensemble Pipeline initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data raw directory: {self.data_raw_dir}")
        logger.info(f"Data processed directory: {self.data_processed_dir}")
        logger.info(f"Outputs directory: {self.outputs_dir}")

    @staticmethod
    def get_default_config():
        """Get default pipeline configuration for all 6 models"""
        return {
            "dea": {
                "input_path": None,  # Will use default
                "years": ["2023", "2022", "2021", "2020", "2019", "2018",
                          "2017", "2016", "2015", "2014", "2013"],
                "output_dir": "data/processed/",
                "output_filename": "dea_results.csv"
            },
            "models": {
                "xgb": {
                    "enabled": True,
                    "output_base_dir": "outputs",
                    "target_column": "CCR_ES",
                    "seed": 42,
                    "epochs": 15,
                    "pop_size": 5
                },
                "gbr": {
                    "enabled": True,
                    "output_base_dir": "outputs",
                    "target_column": "CCR_ES",
                    "seed": 42,
                    "epochs": 15,
                    "pop_size": 5
                },
                "lgbm": {
                    "enabled": True,
                    "output_base_dir": "outputs",
                    "target_column": "CCR_ES",
                    "seed": 42,
                    "epochs": 15,
                    "pop_size": 5
                },
                "rf": {
                    "enabled": True,
                    "output_base_dir": "outputs",
                    "target_column": "CCR_ES",
                    "seed": 42,
                    "epochs": 15,
                    "pop_size": 5
                },
                "et": {
                    "enabled": True,
                    "output_base_dir": "outputs",
                    "target_column": "CCR_ES",
                    "seed": 42,
                    "epochs": 15,
                    "pop_size": 5
                },
                "ada": {
                    "enabled": True,
                    "output_base_dir": "outputs",
                    "target_column": "CCR_ES",
                    "seed": 42,
                    "epochs": 15,
                    "pop_size": 5
                }
            },
            "run_options": {
                "run_dea": True,
                "parallel": False  # Run models in parallel (not recommended for small datasets)
            }
        }

    def check_data_exists(self):
        """Check if required data files exist"""
        excel_path = os.path.join(self.data_raw_dir, "dataset.xlsx")

        if not os.path.exists(excel_path):
            logger.error(f"Excel file not found at: {excel_path}")
            logger.error(f"Please place your dataset.xlsx file in: {self.data_raw_dir}")
            return False

        logger.info(f"Found dataset at: {excel_path}")
        return True

    def run_dea_analysis(self):
        """Run DEA analysis"""
        logger.info("=" * 60)
        logger.info("STEP 1: Running DEA Analysis")
        logger.info("=" * 60)

        dea_config = self.config["dea"]

        try:
            # Check if DEA results already exist
            dea_results_path = os.path.join(self.data_processed_dir, "dea_results.csv")
            if os.path.exists(dea_results_path):
                logger.info(f"DEA results already exist at: {dea_results_path}")
                logger.info("Skipping DEA analysis (use --force-dea to rerun)")
                return None, {"Recommended_Model": "CCR"}  # Default

            dea_results, robustness = run_dea_pipeline(
                input_path=dea_config["input_path"],
                years=dea_config["years"],
                output_dir=self.data_processed_dir
            )

            logger.info("✅ DEA analysis completed successfully")
            logger.info(f"DEA results shape: {dea_results.shape}")
            logger.info(f"Recommended model: {robustness.get('Recommended_Model', 'Unknown')}")

            return dea_results, robustness

        except Exception as e:
            logger.error(f"DEA analysis failed: {e}")
            raise

    def run_model_pipeline(self, model_name: str, model_config: Dict):
        """Run a specific model pipeline"""
        logger.info("=" * 60)
        logger.info(f"Running {model_name.upper()} Ensemble Model")
        logger.info("=" * 60)

        # Set data path to DEA results
        if model_config.get("data_path") is None:
            dea_results_path = os.path.join(self.data_processed_dir, "dea_results.csv")
            model_config["data_path"] = dea_results_path

        # Run the appropriate model
        try:
            if model_name == "xgb":
                results = run_xgb_pipeline(
                    data_path=model_config["data_path"],
                    output_base_dir=os.path.join(self.outputs_dir, "xgb"),
                    target_column=model_config["target_column"],
                    seed=model_config["seed"],
                    epochs=model_config["epochs"],
                    pop_size=model_config["pop_size"]
                )
            elif model_name == "gbr":
                results = run_gbr_pipeline(
                    data_path=model_config["data_path"],
                    output_base_dir=os.path.join(self.outputs_dir, "gbr"),
                    target_column=model_config["target_column"],
                    seed=model_config["seed"],
                    epochs=model_config["epochs"],
                    pop_size=model_config["pop_size"]
                )
            elif model_name == "lgbm":
                results = run_lgbm_pipeline(
                    data_path=model_config["data_path"],
                    output_base_dir=os.path.join(self.outputs_dir, "lgbm"),
                    target_column=model_config["target_column"],
                    seed=model_config["seed"],
                    epochs=model_config["epochs"],
                    pop_size=model_config["pop_size"]
                )
            elif model_name == "rf":
                results = run_rf_pipeline(
                    data_path=model_config["data_path"],
                    output_base_dir=os.path.join(self.outputs_dir, "rf"),
                    target_column=model_config["target_column"],
                    seed=model_config["seed"],
                    epochs=model_config["epochs"],
                    pop_size=model_config["pop_size"]
                )
            elif model_name == "et":
                results = run_et_pipeline(
                    data_path=model_config["data_path"],
                    output_base_dir=os.path.join(self.outputs_dir, "et"),
                    target_column=model_config["target_column"],
                    seed=model_config["seed"],
                    epochs=model_config["epochs"],
                    pop_size=model_config["pop_size"]
                )
            elif model_name == "ada":
                results = run_ada_pipeline(
                    data_path=model_config["data_path"],
                    output_base_dir=os.path.join(self.outputs_dir, "ada"),
                    target_column=model_config["target_column"],
                    seed=model_config["seed"],
                    epochs=model_config["epochs"],
                    pop_size=model_config["pop_size"]
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            logger.info(f"✅ {model_name.upper()} analysis completed successfully")
            logger.info(f"Test R2: {results['test_metrics']['r2']:.4f}")
            logger.info(f"Results saved to: {results['output_dir']}")

            return results

        except Exception as e:
            logger.error(f"{model_name.upper()} analysis failed: {e}")
            raise

    def run_all_models(self):
        """Run all enabled ensemble models"""
        results = {}
        model_configs = self.config["models"]

        for model_name, model_config in model_configs.items():
            if model_config.get("enabled", True):
                try:
                    model_results = self.run_model_pipeline(model_name, model_config)
                    results[model_name] = model_results
                except Exception as e:
                    logger.error(f"Failed to run {model_name}: {e}")
                    # Continue with other models
                    continue
            else:
                logger.info(f"Skipping {model_name.upper()} (disabled in config)")

        return results

    def compare_all_models(self, model_results: Dict):
        """Compare performance of all ensemble models"""
        if len(model_results) < 2:
            logger.info("Not enough models to compare (need at least 2)")
            return None, None

        logger.info("=" * 60)
        logger.info("MODEL COMPARISON - ALL 6 MODELS")
        logger.info("=" * 60)

        comparison = {}
        best_model = None
        best_r2 = -float('inf')

        # Header
        logger.info(f"{'Model':<15} {'R2':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
        logger.info("-" * 60)

        for model_name, results in model_results.items():
            test_metrics = results.get("test_metrics", {})

            comparison[model_name] = {
                "R2": test_metrics.get("r2", 0),
                "RMSE": test_metrics.get("rmse", 0),
                "MAE": test_metrics.get("mae", 0),
                "MAPE": test_metrics.get("mape", 0)
            }

            # Update best model
            r2 = test_metrics.get("r2", -float('inf'))
            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name

            # Log metrics
            logger.info(f"{model_name.upper():<15} {test_metrics.get('r2', 0):<10.4f} "
                        f"{test_metrics.get('rmse', 0):<10.4f} "
                        f"{test_metrics.get('mae', 0):<10.4f} "
                        f"{test_metrics.get('mape', 0):<10.2f}%")

        logger.info(f"\n🏆 Best Model: {best_model.upper()} (R2 = {best_r2:.4f})")

        # Save comparison to file
        self._save_comparison(comparison, best_model)

        return comparison, best_model

    def _save_comparison(self, comparison: Dict, best_model: str):
        """Save model comparison to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.outputs_dir, f"model_comparison_{timestamp}.txt")

        with open(filename, 'w') as file:
            file.write("=" * 60 + "\n")
            file.write("MODEL COMPARISON RESULTS\n")
            file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"{'Model':<15} {'R2':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}\n")
            file.write("-" * 60 + "\n")

            for model_name, metrics in comparison.items():
                file.write(f"{model_name.upper():<15} {metrics['R2']:<10.4f} "
                           f"{metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} "
                           f"{metrics['MAPE']:<10.2f}%\n")

            file.write("\n" + "=" * 60 + "\n")
            file.write(f"BEST MODEL: {best_model.upper()}\n")
            file.write(f"Best R2: {comparison[best_model]['R2']:.4f}\n")
            file.write("=" * 60 + "\n")

        logger.info(f"Model comparison saved to: {filename}")

    def run_full_pipeline(self, force_dea: bool = False):
        """Run the complete DEA-Ensemble pipeline"""
        logger.info("=" * 60)
        logger.info("DEA-ENSEMBLE PIPELINE STARTING (6 MODELS)")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Check if data exists
        if not self.check_data_exists():
            return None

        all_results = {}

        # Run DEA analysis
        if self.config["run_options"]["run_dea"] or force_dea:
            dea_results, robustness = self.run_dea_analysis()
            all_results["dea"] = {
                "results": dea_results,
                "robustness": robustness
            }
        else:
            logger.info("Skipping DEA analysis (disabled in config)")

        # Run all ensemble models
        model_results = self.run_all_models()
        all_results["models"] = model_results

        # Compare models if at least 2 were successful
        successful_models = {k: v for k, v in model_results.items() if v is not None}
        if len(successful_models) >= 2:
            comparison, best_model = self.compare_all_models(successful_models)
            all_results["comparison"] = comparison
            all_results["best_model"] = best_model
        else:
            logger.warning(
                f"Only {len(successful_models)} model(s) completed successfully. Need at least 2 for comparison.")

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Total duration: {duration}")
        logger.info(f"Models completed: {len(successful_models)}/6")
        logger.info("=" * 60)

        # Save summary
        self._save_pipeline_summary(all_results, duration)

        return all_results

    def _save_pipeline_summary(self, results: Dict, duration):
        """Save pipeline summary to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.outputs_dir, f"pipeline_summary_{timestamp}.txt")

        with open(filename, 'w') as file:
            file.write("=" * 60 + "\n")
            file.write("DEA-ENSEMBLE PIPELINE SUMMARY\n")
            file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Duration: {duration}\n")
            file.write("=" * 60 + "\n\n")

            # DEA info
            if "dea" in results:
                file.write("DEA ANALYSIS:\n")
                file.write(f"  Results file: data/processed/dea_results.csv\n")
                if results["dea"].get("robustness"):
                    file.write(
                        f"  Recommended model: {results['dea']['robustness'].get('Recommended_Model', 'Unknown')}\n")

            # Models info
            if "models" in results:
                file.write("\nENSEMBLE MODELS:\n")
                for model_name, model_results in results["models"].items():
                    if model_results:
                        file.write(f"  {model_name.upper():<6}: ")
                        file.write(f"R2={model_results.get('test_metrics', {}).get('r2', 0):.4f} | ")
                        file.write(f"Output: {model_results.get('output_dir', 'N/A')}\n")

            # Comparison info
            if "comparison" in results and "best_model" in results:
                file.write("\nMODEL COMPARISON:\n")
                file.write(f"  Best model: {results['best_model'].upper()}\n")
                file.write(f"  Best R2: {results['comparison'][results['best_model']]['R2']:.4f}\n")

            file.write("\n" + "=" * 60 + "\n")
            file.write("FILES CREATED:\n")
            file.write("=" * 60 + "\n")

            # List output directories
            if "models" in results:
                for model_name, model_results in results["models"].items():
                    if model_results:
                        file.write(f"{model_name.upper()}: {model_results.get('output_dir', 'N/A')}\n")

        logger.info(f"Pipeline summary saved to: {filename}")


def main():
    """Main entry point for the pipeline"""
    parser = argparse.ArgumentParser(description="DEA-Ensemble Pipeline with 6 Models")
    parser.add_argument("--no-dea", action="store_true", help="Skip DEA analysis")
    parser.add_argument("--force-dea", action="store_true", help="Force rerun DEA analysis")
    parser.add_argument("--models", nargs="+",
                        choices=["xgb", "gbr", "lgbm", "rf", "et", "ada", "all"],
                        default=["all"],
                        help="Models to run (default: all)")
    parser.add_argument("--skip-models", nargs="+",
                        choices=["xgb", "gbr", "lgbm", "rf", "et", "ada"],
                        help="Models to skip")

    args = parser.parse_args()

    # Create config
    config = DEAEnsemblePipeline.get_default_config()

    # Update config based on command line arguments
    if args.no_dea:
        config["run_options"]["run_dea"] = False

    # Handle model selection
    if "all" not in args.models:
        # Disable all models first
        for model_name in config["models"]:
            config["models"][model_name]["enabled"] = False

        # Enable selected models
        for model_name in args.models:
            if model_name in config["models"]:
                config["models"][model_name]["enabled"] = True

    # Handle skip models
    if args.skip_models:
        for model_name in args.skip_models:
            if model_name in config["models"]:
                config["models"][model_name]["enabled"] = False

    # Run pipeline
    pipeline = DEAEnsemblePipeline(config)
    results = pipeline.run_full_pipeline(force_dea=args.force_dea)

    if results:
        logger.info("✅ Pipeline completed successfully!")
        logger.info("Run compare_models.py for detailed comparison")
    else:
        logger.error("❌ Pipeline failed!")


if __name__ == "__main__":
    main()