from bounds import bounds
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

class CooperStandard:
    """
    A class to handle loading, preprocessing, categorizing, normalizing, and splitting batch process data.
    """
    def __init__(self, file_path, sheet_names, variable_names):
        self.file_path = file_path
        self.sheet_names = sheet_names
        self.variable_names = variable_names
        self.df = None
        self.scaler_params = None  # To store min/max for scalar and time series features

    def load_data(self):
        try:
            self.df = pd.concat(
                [pd.read_excel(self.file_path, sheet_name=sheet, usecols=self.variable_names, dtype=str)
                 for sheet in self.sheet_names],
                ignore_index=True
            )
            for col in ["MDRTorqueS1", "MDRTorqueS2"]:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            self.df = self.df.apply(pd.to_numeric, errors='ignore')
            for col in ["start_time", "end_time"]:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file '{self.file_path}' was not found.")
        except ValueError as e:
            raise ValueError(f"Error loading sheets: {e}")
        return self.df

    def _check_columns(self, required_cols):
        missing_columns = [col for col in required_cols if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataset: {missing_columns}")

    def df_variables(self):
        if self.df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        self._check_columns(self.variable_names)
        return self.df[self.variable_names]

    def preprocessing(self, length_threshold=300):
        if self.df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        df_vars = self.df_variables().dropna()
        df_vars["end-start"] = (df_vars["end_time"] - df_vars["start_time"]).dt.total_seconds()
        df_vars.drop(columns=["start_time", "end_time"], inplace=True)
        mask = (df_vars["MDRTorqueS1"].str.len() >= length_threshold) & \
               (df_vars["MDRTorqueS2"].str.len() >= length_threshold)
        removed_batches = {
            row["batch_number"]: {"len(S1)=len(S2)": len(row["MDRTorqueS1"])}
            for _, row in df_vars[~mask].iterrows()
        }
        return df_vars[mask].reset_index(drop=True), removed_batches

    # def categorize_t5(self, lb, ub):
    #     df_cleaned, _ = self.preprocessing()
    #     self._check_columns(["t5"])
    #     df_low = df_cleaned[df_cleaned["t5"] < lb]
    #     df_normal = df_cleaned[(df_cleaned["t5"] >= lb) & (df_cleaned["t5"] <= ub)]
    #     df_high = df_cleaned[df_cleaned["t5"] > ub]
    #     return df_low, df_normal, df_high

    def convert_to_dict(self, df):
        batch_dict = {}
        # Find the minimum time series length across all batches
        min_length = min(len(row["MDRTorqueS1"]) for _, row in df.iterrows())
        
        for _, row in df.iterrows():
            batch_number = row["batch_number"]
            scalar_features = [
                row["mh"], row["ml"], row["TimeAtML"], row["TimeAtML_min"],
                row["ml_min"], row["end-start"]
            ]
            # Extract truncated time series data
            time_values = [t[0] for t in row["MDRTorqueS1"][:min_length]]
            S1_values = [s[1] for s in row["MDRTorqueS1"][:min_length]]
            S2_values = [s[1] for s in row["MDRTorqueS2"][:min_length]]
            
            temp_df = pd.DataFrame({"time": time_values, "S1": S1_values, "S2": S2_values})
            temp_df = temp_df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
            
            if temp_df.isnull().values.any():
                raise ValueError(f"NaN values remain in time series for batch {batch_number} after interpolation.")
            
            S1_S2_ratio = temp_df["S1"] / (temp_df["S2"] + 1e-8)
            time_series = np.column_stack((temp_df["time"], temp_df["S1"], temp_df["S2"], S1_S2_ratio))
            
            batch_dict[batch_number] = {
                "scalar_features": scalar_features,
                "time_series": time_series,
                "t5": row["t5"]
            }
        return batch_dict

    def normalize_data(self, data_dict, fit=False):
        """
        Normalizes scalar features and time series using min-max scaling.
        
        Args:
            data_dict (dict): Dictionary from convert_to_dict with batch data.
            fit (bool): If True, compute min/max from this data; if False, use stored scaler_params.
        
        Returns:
            dict: Normalized data dictionary.
        """
        if fit:
            # Compute min/max for scalar features and time series from training data
            scalar_values = np.array([d["scalar_features"] for d in data_dict.values()])
            time_series_values = np.concatenate([d["time_series"] for d in data_dict.values()], axis=0)

            self.scaler_params = {
                "scalar_min": scalar_values.min(axis=0),
                "scalar_max": scalar_values.max(axis=0),
                "ts_min": time_series_values.min(axis=0),
                "ts_max": time_series_values.max(axis=0)
            }

        if self.scaler_params is None:
            raise ValueError("Scaler parameters not initialized. Run with fit=True on training data first.")

        # Normalize the data
        normalized_dict = {}
        for batch_number, data in data_dict.items():
            # Normalize scalar features
            scalar_norm = (np.array(data["scalar_features"]) - self.scaler_params["scalar_min"]) / \
                          (self.scaler_params["scalar_max"] - self.scaler_params["scalar_min"] + 1e-8)
            
            # Normalize time series
            ts_norm = (data["time_series"] - self.scaler_params["ts_min"]) / \
                      (self.scaler_params["ts_max"] - self.scaler_params["ts_min"] + 1e-8)
            
            normalized_dict[batch_number] = {
                "scalar_features": scalar_norm.tolist(),
                "time_series": ts_norm,
                "t5": data["t5"]  # Leave t5 unnormalized (can be normalized separately if needed)
            }
        return normalized_dict

    def split_data(self, data_dict, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
        """
        Splits the data dictionary into train, validation, and test sets.
        
        Args:
            data_dict (dict): Dictionary from convert_to_dict.
            train_size (float): Proportion of data for training.
            val_size (float): Proportion of data for validation.
            test_size (float): Proportion of data for testing.
            random_state (int): Seed for reproducibility.
        
        Returns:
            tuple: (train_dict, val_dict, test_dict)
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        batch_numbers = list(data_dict.keys())
        train_batches, temp_batches = train_test_split(
            batch_numbers, train_size=train_size, random_state=random_state
        )
        val_relative_size = val_size / (val_size + test_size)
        val_batches, test_batches = train_test_split(
            temp_batches, train_size=val_relative_size, random_state=random_state
        )

        train_dict = {k: data_dict[k] for k in train_batches}
        val_dict = {k: data_dict[k] for k in val_batches}
        test_dict = {k: data_dict[k] for k in test_batches}

        return train_dict, val_dict, test_dict
    
    def print_t5_categories(self, data_dict, lb, ub):
        # Initialize counters for each category
        low_count = 0
        normal_count = 0
        high_count = 0
        
        # Iterate through the dictionary values to access each batch
        for batch in data_dict.values():
            t5 = batch['t5']  # Extract the 't5' value
            
            # Categorize based on t5 value
            if t5 < lb:
                low_count += 1
            elif lb <= t5 <= ub:
                normal_count += 1
            else:  # t5 > ub
                high_count += 1
        
        # Calculate total number of batches
        total = len(data_dict)
        
        # Print the results
        print(f"Total number of batches: {total}")
        print(f"Number of low batches (t5 < {lb}): {low_count}")
        print(f"Number of normal batches ({lb} <= t5 <= {ub}): {normal_count}")
        print(f"Number of high batches (t5 > {ub}): {high_count}\n")