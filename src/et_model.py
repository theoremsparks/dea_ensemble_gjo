import os
import datetime
import random
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, Optional

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import ExtraTreesRegressor

import shap
from mealpy.swarm_based import GJO
from mealpy import FloatVar, IntegerVar

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ETModelTrainer:
    def __init__(self, data_path: str = None, output_base_dir: str = "outputs", seed: int = 42):

        self.seed = seed
        self.set_global_seed(seed)

        # Set data path
        if data_path is None:
            # Go up one level from src/ to project root, then to data/processed/
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(project_root, "data", "processed", "dea_results.csv")

        self.data_path = data_path
        self.output_base_dir = output_base_dir

        # Create timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.readable_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create specific output directory for this run
        self.output_dir = os.path.join(output_base_dir, f"et_gjo_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize attributes
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.model = None
        self.best_params = None
        self.gjo = None
        self.g_best = None

        logger.info(f"ETModelTrainer initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Random seed: {self.seed}")

    def set_global_seed(self, seed: int = 42):
        """Set seeds for Python, NumPy, and hash-based operations."""
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def smape(self, y_true, y_pred):
        """Calculate symmetric mean absolute percentage error (sMAPE)"""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return 100 * np.mean(numerator / denominator)

    def load_and_prepare_data(self, target_column: str = "CCR_ES"):
        logger.info(f"Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path, keep_default_na=False)

        # Define columns to drop
        columns_to_drop = ["DMU", "Year", "CCR_ES", "BCC_ES", "Scale_Efficiency",
                           "RTS", "index", "Ticker", "Company"]

        # Keep only columns that exist in the data
        existing_columns = [col for col in columns_to_drop if col in self.data.columns]

        # Features (X)
        self.X = self.data.drop(columns=existing_columns).values
        self.feature_names = self.data.drop(columns=existing_columns).columns.tolist()

        # Target (y) - using specified column
        if target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data. Available columns: {list(self.data.columns)}")

        self.y = self.data[target_column].values

        logger.info(f"Data loaded: {len(self.data)} samples, {len(self.feature_names)} features")
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Target: {target_column}")

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.15, random_state=self.seed
        )

        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def optimize_hyperparameters(self, epochs: int = 5, pop_size: int = 5):

        logger.info("Starting hyperparameter optimization with GJO...")

        # Declare search space for Extra Trees Regressor
        bounds = [
            IntegerVar(lb=100, ub=1000, name="n_estimators"),
            IntegerVar(lb=4, ub=10, name="max_depth"),
            IntegerVar(lb=3, ub=10, name="min_samples_split"),
            IntegerVar(lb=2, ub=10, name="min_samples_leaf"),
            FloatVar(lb=0.1, ub=1.0, name="max_features"),
            FloatVar(lb=0.0, ub=0.5, name="min_impurity_decrease")
        ]

        # Fitness function (5-fold CV RMSE on train data)
        def cv_rmse(solution):
            (n_estimators, max_depth, min_samples_split, min_samples_leaf,
             max_features, min_impurity_decrease) = solution

            params = {
                "n_estimators": int(n_estimators),
                "max_depth": None if int(max_depth) == 10 else int(max_depth),  # None means unlimited
                "min_samples_split": int(min_samples_split),
                "min_samples_leaf": int(min_samples_leaf),
                "max_features": float(max_features),
                "min_impurity_decrease": float(min_impurity_decrease),
                "random_state": self.seed,
                "n_jobs": -1
            }

            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            fold_rmse = []

            for tr_idx, val_idx in kf.split(self.X_train):
                model = ExtraTreesRegressor(**params)
                model.fit(self.X_train[tr_idx], self.y_train[tr_idx])
                preds = model.predict(self.X_train[val_idx])
                fold_rmse.append(np.sqrt(mean_squared_error(self.y_train[val_idx], preds)))

            return np.mean(fold_rmse)  # GJO minimizes this

        # Define optimization problem
        problem = {"obj_func": cv_rmse, "bounds": bounds, "minmax": "min"}

        # Run Golden Jackal Optimizer
        self.gjo = GJO.OriginalGJO(epoch=epochs, pop_size=pop_size, seed=self.seed)

        tic = time.time()
        self.g_best = self.gjo.solve(problem, mode="swarm")
        toc = time.time()

        logger.info(f"GJO optimization completed in {toc - tic:.1f} seconds")
        logger.info(f"Best CV-RMSE: {self.g_best.target.fitness:.6f}")

        # Extract best parameters
        self.best_params = {
            "n_estimators": int(self.g_best.solution[0]),
            "max_depth": None if int(self.g_best.solution[1]) == 10 else int(self.g_best.solution[1]),
            "min_samples_split": int(self.g_best.solution[2]),
            "min_samples_leaf": int(self.g_best.solution[3]),
            "max_features": self.g_best.solution[4],
            "min_impurity_decrease": self.g_best.solution[5],
            "random_state": self.seed,
            "n_jobs": -1
        }

        logger.info("Best hyperparameters found:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")

        return self.best_params


    def train_final_model(self):
        """Train final Extra Trees model with optimized hyperparameters"""
        logger.info("Training final Extra Trees model...")

        self.model = ExtraTreesRegressor(**self.best_params)
        self.model.fit(self.X_train, self.y_train)

        # Save the trained model
        model_path = os.path.join(self.output_dir, f"et_model_{self.timestamp}.pkl")
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to: {model_path}")

        return self.model

    def evaluate_model(self):
        """Evaluate model on train and test sets"""
        logger.info("Evaluating model performance...")

        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_train_pred = np.round(y_train_pred, 4)

        y_test_pred = self.model.predict(self.X_test)
        y_test_pred = np.round(y_test_pred, 4)

        # Calculate metrics
        train_metrics = self._calculate_metrics(self.y_train, y_train_pred, "Train")
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred, "Test")

        # Save predictions
        self._save_predictions(self.y_test, y_test_pred, "test")
        self._save_predictions(self.y_train, y_train_pred, "train")

        # Save metrics to file
        self._save_metrics(train_metrics, test_metrics)

        return train_metrics, test_metrics, y_train_pred, y_test_pred

    def _calculate_metrics(self, y_true, y_pred, dataset_name: str):
        """Calculate evaluation metrics"""
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "smape": self.smape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "r": np.corrcoef(y_true, y_pred)[0, 1]
        }

        logger.info(f"\n{dataset_name} metrics:")
        for name, value in metrics.items():
            logger.info(f"{name.upper():<6}: {value:.4f}" if name != 'mape' and name != 'smape'
                        else f"{name.upper():<6}: {value:.2f}%")

        return metrics

    def _save_predictions(self, y_true, y_pred, dataset_type: str):
        """Save predictions to CSV"""
        predictions_df = pd.DataFrame({
            "Actual": y_true,
            "ET_Predicted": y_pred
        })

        filename = os.path.join(self.output_dir, f"et_{dataset_type}_predictions_{self.timestamp}.csv")
        predictions_df.to_csv(filename, index=False)
        logger.info(f"{dataset_type.capitalize()} predictions saved to: {filename}")

    def _save_metrics(self, train_metrics: Dict, test_metrics: Dict):
        """Save metrics to text file"""
        filename = os.path.join(self.output_dir, f"et_metrics_{self.timestamp}.txt")

        with open(filename, 'w') as file:
            file.write("=" * 50 + "\n")
            file.write(f"Extra Trees Model Training Results\n")
            file.write(f"Run Timestamp: {self.readable_time}\n")
            file.write("=" * 50 + "\n\n")

            file.write("Hyperparameters:\n")
            for key, value in self.best_params.items():
                file.write(f"  {key}: {value}\n")

            file.write("\n" + "=" * 30 + "\n")
            file.write("Training Data Metrics:\n")
            file.write("=" * 30 + "\n")
            for name, value in train_metrics.items():
                if name in ['mape', 'smape']:
                    file.write(f"{name.upper():<8}: {value:.2f}%\n")
                else:
                    file.write(f"{name.upper():<8}: {value:.4f}\n")

            file.write("\n" + "=" * 30 + "\n")
            file.write("Testing Data Metrics:\n")
            file.write("=" * 30 + "\n")
            for name, value in test_metrics.items():
                if name in ['mape', 'smape']:
                    file.write(f"{name.upper():<8}: {value:.2f}%\n")
                else:
                    file.write(f"{name.upper():<8}: {value:.4f}\n")

        logger.info(f"Metrics saved to: {filename}")

    def plot_feature_importance(self):
        """Plot and save feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_final_model() first.")

        feature_importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title('Ranking of Features (Extra Trees)', fontsize=16)
        plt.xlabel('Degree of Influence', fontsize=14)
        plt.ylabel('Feature Names', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.output_dir, f"feature_importance_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance plot saved to: {plot_path}")

        # Save importance data
        csv_path = os.path.join(self.output_dir, f"feature_importance_{self.timestamp}.csv")
        importance_df.to_csv(csv_path, index=False)

        return importance_df

    def plot_scatter_with_fit(self, y_true, y_pred, dataset_type: str):
        """Create scatter plot with best fit line"""
        # Axis limits with padding
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        xmin, xmax = lo - pad, hi + pad

        # Best-fit line
        a, b = np.polyfit(y_true, y_pred, 1)
        x_line = np.linspace(xmin, xmax, 200)
        y_line = a * x_line + b

        # Correlation coefficient
        R = np.corrcoef(y_true, y_pred)[0, 1]

        # Create plot
        plt.figure(figsize=(7.2, 6.4))
        plt.scatter(y_true, y_pred, alpha=0.65, edgecolor='k', linewidth=0.5)

        # Ideal line
        plt.plot([xmin, xmax], [xmin, xmax],
                 linestyle='--', linewidth=2, color='red', label='Ideal: y = x')

        # Best-fit line
        plt.plot(x_line, y_line, linewidth=2, color='green', label='Best fit')

        # Labels
        plt.title(f'{dataset_type} Set: Predicted vs Actual (Extra Trees)', fontsize=16)
        plt.xlabel('True Efficiency Score', fontsize=14)
        plt.ylabel('Predicted Efficiency Score', fontsize=14)
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, xmax)
        plt.legend(fontsize=12, loc='lower right')

        # Equation text
        eq_text = f'y = {a:.3f}x + {b:.3f}\nR = {R:.4f}'
        plt.gca().text(
            0.02, 0.98, eq_text,
            transform=plt.gca().transAxes,
            ha='left', va='top',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.8, linewidth=0.5)
        )

        plt.tight_layout()

        # Save plot
        outpath = os.path.join(self.output_dir, f"scatter_{dataset_type.lower()}_{self.timestamp}.png")
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Scatter plot for {dataset_type} saved to: {outpath}")

        return outpath

    def perform_shap_analysis(self):
        """Perform SHAP analysis on test set"""
        logger.info("Running SHAP analysis...")

        # Create explainer
        explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(self.X_test)

        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        summary_path = os.path.join(self.output_dir, f"shap_summary_{self.timestamp}.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary plot saved to: {summary_path}")

        # Bar plot
        plt.figure()
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names,
                          plot_type="bar", show=False)
        plt.tight_layout()
        bar_path = os.path.join(self.output_dir, f"shap_bar_{self.timestamp}.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP bar plot saved to: {bar_path}")

        # Dependence plots for top features
        importance_df = self.plot_feature_importance()
        top_features = importance_df['Feature'].values[:5]

        for feature in top_features:
            plt.figure()
            shap.dependence_plot(feature, shap_values, self.X_test,
                                 feature_names=self.feature_names,
                                 interaction_index=None, show=False)
            plt.title(f"SHAP Dependence Plot: {feature}", fontsize=14)
            plt.tight_layout()
            dep_path = os.path.join(self.output_dir, f"shap_dependence_{feature}_{self.timestamp}.png")
            plt.savefig(dep_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP dependence plot for {feature} saved to: {dep_path}")

        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{name}" for name in self.feature_names])
        shap_csv_path = os.path.join(self.output_dir, f"shap_values_{self.timestamp}.csv")
        shap_df.to_csv(shap_csv_path, index=False)
        logger.info(f"SHAP values saved to: {shap_csv_path}")

        return shap_values, explainer

    def run_full_pipeline(self, target_column: str = "CCR_ES", epochs: int = 5, pop_size: int = 5):

        logger.info("Starting Extra Trees full pipeline...")

        # 1. Load and prepare data
        self.load_and_prepare_data(target_column)

        # 2. Optimize hyperparameters
        self.optimize_hyperparameters(epochs, pop_size)

        # 4. Train final model
        self.train_final_model()

        # 5. Evaluate model
        train_metrics, test_metrics, y_train_pred, y_test_pred = self.evaluate_model()

        # 6. Create visualizations
        self.plot_feature_importance()
        self.plot_scatter_with_fit(self.y_train, y_train_pred, "Train")
        self.plot_scatter_with_fit(self.y_test, y_test_pred, "Test")

        # 7. SHAP analysis
        shap_values, explainer = self.perform_shap_analysis()

        logger.info("✅ Extra Trees pipeline completed!")

        results = {
            "model": self.model,
            "best_params": self.best_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "output_dir": self.output_dir,
            "shap_values": shap_values,
            "explainer": explainer
        }

        return results


def run_et_pipeline(data_path: str = None, output_base_dir: str = "outputs",
                    target_column: str = "CCR_ES", seed: int = 42,
                    epochs: int = 5, pop_size: int = 5):

    trainer = ETModelTrainer(data_path, output_base_dir, seed)
    return trainer.run_full_pipeline(target_column, epochs, pop_size)


if __name__ == "__main__":
    # Example usage
    results = run_et_pipeline(
        data_path=None,  # Will use default from data/processed/
        output_base_dir="outputs",
        target_column="CCR_ES",
        seed=42,
        epochs=5,
        pop_size=5
    )

    print(f"\n✅ Extra Trees pipeline completed!")
    print(f"Results saved to: {results['output_dir']}")
    print(f"Test R2: {results['test_metrics']['r2']:.4f}")