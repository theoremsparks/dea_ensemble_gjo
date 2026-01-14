import pulp
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DEAAnalyzer:
    def __init__(self, file_path=None, years=None, output_dir="data/processed/"):
        # Set default file path if not provided
        if file_path is None:
            # Go up one level from src/ to project root, then to data/raw/
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(project_root, "data", "raw", "dataset.xlsx")

        self.file_path = file_path
        self.years = years or ["2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013"]
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Define columns
        self.input_cols = ["Total Asset", "Total Equity", "Operating Expenses"]
        self.output_cols = ["Revenue", "Net Income", "Operating Income"]
        self.noise_cols = self.input_cols + self.output_cols

        logger.info(f"DEA Analyzer initialized")
        logger.info(f"Input file: {self.file_path}")
        logger.info(f"Years: {self.years}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Input columns: {self.input_cols}")
        logger.info(f"Output columns: {self.output_cols}")

    def classify_rts(self, ccr, bcc, tol=1e-4):
        """Classify returns to scale"""
        if abs(ccr - bcc) <= tol:
            return "CRS"
        return "IRS" if ccr < bcc else "DRS"

    def run_dea_analysis(self, data, year, dmu_start_index=0, model_type='CCR'):
        """Run CCR or BCC DEA analysis"""
        solver = pulp.PULP_CBC_CMD(msg=False, options=["primalTolerance=1e-9", "dualTolerance=1e-9"])
        inputs = data[self.input_cols].values
        outputs = data[self.output_cols].values

        results = []
        for i in range(len(data)):
            prob = pulp.LpProblem(f"{model_type}_DEA_DMU_{i + 1}_{year}", pulp.LpMaximize)
            num_vars = inputs.shape[1] + outputs.shape[1]
            weights = pulp.LpVariable.dicts("Weight", range(num_vars), lowBound=0)
            input_weights = [weights[k] for k in range(inputs.shape[1])]
            output_weights = [weights[k + inputs.shape[1]] for k in range(outputs.shape[1])]

            # Objective function
            if model_type == 'CCR':
                prob += pulp.lpSum(output_weights[j] * outputs[i][j] for j in range(outputs.shape[1]))
            else:  # BCC
                u0 = pulp.LpVariable("u0", lowBound=None)
                prob += pulp.lpSum(output_weights[j] * outputs[i][j] for j in range(outputs.shape[1])) + u0

            prob += pulp.lpSum(input_weights[k] * inputs[i][k] for k in range(inputs.shape[1])) == 1

            # Efficiency constraints
            for dmu in range(len(data)):
                constraint = pulp.lpSum(output_weights[j] * outputs[dmu][j] for j in range(outputs.shape[1]))
                if model_type == 'BCC':
                    constraint += u0
                prob += constraint <= pulp.lpSum(input_weights[k] * inputs[dmu][k] for k in range(inputs.shape[1]))

            prob.solve(solver)
            score_key = f"{model_type}_ES"
            results.append({
                'DMU': dmu_start_index + i + 1,
                'Year': year,
                score_key: round(pulp.value(prob.objective), 4)
            })

        return pd.DataFrame(results)

    def add_random_noise(self, data, noise_level=0.2):
        """Add random noise varying across both rows AND columns"""
        noisy_data = data.copy()
        logger.debug(f"Adding ±{noise_level * 100}% noise to {len(data)} companies...")

        # Create a noise matrix with different random values for each cell
        noise_matrix = np.random.uniform(
            low=-noise_level,
            high=noise_level,
            size=(len(data), len(self.noise_cols))
        )

        for row_idx in range(len(noisy_data)):
            # Randomly select which columns to perturb for this row
            n_cols_to_perturb = np.random.randint(1, len(self.noise_cols) + 1)
            cols_to_perturb = np.random.choice(
                range(len(self.noise_cols)),
                size=n_cols_to_perturb,
                replace=False
            )

            for col_idx in cols_to_perturb:
                col_name = self.noise_cols[col_idx]
                original_val = noisy_data.iloc[row_idx][col_name]

                # Apply unique noise for this specific cell
                cell_noise = noise_matrix[row_idx, col_idx]
                noisy_value = original_val * (1 + cell_noise)

                # Ensure reasonable bounds
                if noisy_value < original_val * 0.1:  # Don't reduce below 10% of original
                    noisy_value = original_val * 0.1
                if noisy_value > original_val * 10:  # Don't increase above 10x original
                    noisy_value = original_val * 10

                noisy_data.iloc[row_idx, noisy_data.columns.get_loc(col_name)] = noisy_value

        return noisy_data

    def robustness_test(self, seed=42):
        """Perform robustness test on 10% sample"""
        np.random.seed(seed)  # Fixed seed for reproducible results

        # Combine all years data
        all_data = pd.concat([pd.read_excel(self.file_path, sheet_name=year).assign(Year=year)
                              for year in self.years], ignore_index=True)

        logger.info(f"Combined data: {len(all_data)} companies")

        # Select 10% sample
        n_sample = max(1, int(len(all_data) * 0.10))
        sample_indices = np.random.choice(len(all_data), size=n_sample, replace=False)
        test_sample = all_data.iloc[sample_indices].reset_index(drop=True)

        logger.info(f"Robustness test sample: {n_sample} companies")

        # Run models on original and noisy data
        results = {}
        for condition in ['original', 'noisy']:
            data = test_sample if condition == 'original' else self.add_random_noise(test_sample)
            results[f'ccr_{condition}'] = self.run_dea_analysis(data, "Robustness_Test", 0, 'CCR')
            results[f'bcc_{condition}'] = self.run_dea_analysis(data, "Robustness_Test", 0, 'BCC')

        # Calculate correlations
        ccr_corr = spearmanr(results['ccr_original']['CCR_ES'], results['ccr_noisy']['CCR_ES'])[0]
        bcc_corr = spearmanr(results['bcc_original']['BCC_ES'], results['bcc_noisy']['BCC_ES'])[0]

        recommended = 'CCR' if ccr_corr > bcc_corr else 'BCC'

        logger.info(f"Robustness Results:")
        logger.info(f"CCR Correlation: {ccr_corr:.6f}, BCC Correlation: {bcc_corr:.6f}")
        logger.info(f"Recommended Model: {recommended}")

        return {
            'Recommended_Model': recommended,
            'CCR_Correlation': ccr_corr,
            'BCC_Correlation': bcc_corr
        }

    def run_full_analysis(self, output_filename="dea_results.csv"):
        """Run complete DEA analysis with robustness test"""
        logger.info("=" * 60)
        logger.info("DEA ANALYSIS WITH ROBUSTNESS TEST")
        logger.info("=" * 60)

        # Step 1: Robustness test
        robustness_result = self.robustness_test()
        recommended_model = robustness_result['Recommended_Model']

        # Save robustness results
        robustness_df = pd.DataFrame([robustness_result])
        robustness_path = os.path.join(self.output_dir, "robustness_results.csv")
        robustness_df.to_csv(robustness_path, index=False)
        logger.info(f"Robustness results saved to: {robustness_path}")

        # Step 2: Full yearly analysis
        all_results, current_dmu = [], 0

        for year in self.years:
            logger.info(f"Processing {year}...")
            data = pd.read_excel(self.file_path, sheet_name=year, keep_default_na=False).reset_index(drop=True)

            # Run both models for RTS classification
            ccr_results = self.run_dea_analysis(data, year, current_dmu, 'CCR')
            bcc_results = self.run_dea_analysis(data, year, current_dmu, 'BCC')

            # Merge results
            merged = ccr_results.merge(bcc_results, on=["DMU", "Year"])
            merged["Scale_Efficiency"] = (merged["CCR_ES"] / merged["BCC_ES"]).round(4)
            merged["RTS"] = merged.apply(lambda row: self.classify_rts(row["CCR_ES"], row["BCC_ES"]), axis=1)

            # Add company info
            company_info = data[["Company", "Ticker"] + self.input_cols + self.output_cols].reset_index()
            company_info["DMU"] = company_info.index + current_dmu + 1
            merged = merged.merge(company_info, on="DMU")

            all_results.append(merged)
            current_dmu += len(data)
            logger.info(f"  {year}: {len(data)} companies processed")

        # Combine and save results
        final_results = pd.concat(all_results, ignore_index=True)

        # Save to output directory
        output_path = os.path.join(self.output_dir, output_filename)
        final_results.to_csv(output_path, index=False)

        logger.info(f"\n✅ DEA Analysis Completed!")
        logger.info(f"Total DMUs: {len(final_results)}")
        logger.info(f"Recommended Model: {recommended_model}")
        logger.info(f"Results saved to: {output_path}")

        return final_results, robustness_result


def run_dea_pipeline(input_path=None, years=None, output_dir="data/processed/"):
    """
    Convenience function to run the full DEA pipeline
    """
    analyzer = DEAAnalyzer(input_path, years, output_dir)
    return analyzer.run_full_analysis()


if __name__ == "__main__":
    # Run DEA analysis
    analyzer = DEAAnalyzer()
    results, robustness = analyzer.run_full_analysis()

    print(f"DEA analysis complete. Results saved to: {analyzer.output_dir}")