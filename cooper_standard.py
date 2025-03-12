from bounds import bounds
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

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
        df_vars = self.df_variables().dropna().copy()
        df_vars["end-start"] = (df_vars["end_time"] - df_vars["start_time"]).dt.total_seconds()
        df_vars.drop(columns=["start_time", "end_time"], inplace=True)
        mask = (df_vars["MDRTorqueS1"].str.len() >= length_threshold) & \
               (df_vars["MDRTorqueS2"].str.len() >= length_threshold)
        removed_batches = {
            row["batch_number"]: {"len(S1)=len(S2)": len(row["MDRTorqueS1"])}
            for _, row in df_vars[~mask].iterrows()
        }
        return df_vars[mask].reset_index(drop=True), removed_batches

    def convert_to_dict(self, df, fixed_length=None):
        batch_dict = {}
        # If fixed_length is provided, use it; otherwise compute the minimum from this df.
        if fixed_length is None:
            min_length = min(
                min(len(row["MDRTorqueS1"]), len(row["MDRTorqueS2"])) for _, row in df.iterrows()
            )
        else:
            min_length = fixed_length

        for _, row in df.iterrows():
            batch_number = row["batch_number"]
            scalar_features = [
                row["mh"], row["ml"], row["TimeAtML"], row["TimeAtML_min"],
                row["ml_min"], row["end-start"]
            ]
            # Truncate both time series to the determined min_length.
            ts1_truncated = row["MDRTorqueS1"][:min_length]
            ts2_truncated = row["MDRTorqueS2"][:min_length]
            
            time_values = [pair[0] for pair in ts1_truncated]
            S1_values = [pair[1] for pair in ts1_truncated]
            S2_values = [pair[1] for pair in ts2_truncated]
            
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
    
    def balance_data_with_synthetic(self, df, lb, ub, k=5):
        """
        Generate synthetic data for minority classes (t5 < lb and t5 > ub)
        using a SMOTE-like approach that takes into account the interactions among 
        scalar features and time-series summary statistics.
        
        This method first labels each row as "low", "normal", or "high" based on t5,
        determines the number of samples needed to match the majority region, and then 
        generates synthetic samples via interpolation between k-nearest neighbors.
        
        Args:
            df (pd.DataFrame): DataFrame after preprocessing.
            lb (float): Lower bound for t5.
            ub (float): Upper bound for t5.
            k (int): Number of neighbors to consider.
        
        Returns:
            pd.DataFrame: The augmented DataFrame including synthetic samples.
        """
        df_aug = df.copy()
        # Label each row with its region.
        df_aug["region"] = df_aug["t5"].apply(lambda x: "low" if x < lb else ("high" if x > ub else "normal"))
        region_counts = df_aug["region"].value_counts()
        target_count = region_counts.max()
        synthetic_rows = []
        
        # Define scalar columns used later (including the target t5 in the NN feature vector)
        scalar_cols = ["mh", "ml", "TimeAtML", "TimeAtML_min", "ml_min", "end-start"]
        scalar_cols_for_nn = scalar_cols + ["t5"]
        ts_cols = ["MDRTorqueS1", "MDRTorqueS2"]
        
        # Function to compute a simple summary statistic (mean measurement) for a time series.
        def ts_summary(row, col):
            ts_list = row[col]
            arr = np.array(ts_list)
            return arr[:, 1].mean() if len(arr) > 0 else 0
        
        # Build a feature vector combining scalar features, t5, and time-series summaries.
        def build_feature_vector(row):
            scalar_feats = [float(row[col]) for col in scalar_cols_for_nn]
            ts_feats = [ts_summary(row, col) for col in ts_cols]
            return np.array(scalar_feats + ts_feats)
        
        from sklearn.neighbors import NearestNeighbors
        # Process only the minority regions ("low" and "high").
        for region in ["low", "high"]:
            region_df = df_aug[df_aug["region"] == region].reset_index(drop=True)
            n_current = len(region_df)
            n_to_generate = target_count - n_current
            if n_to_generate <= 0:
                continue
            
            features = region_df.apply(build_feature_vector, axis=1).tolist()
            features = np.vstack(features)
            
            nn = NearestNeighbors(n_neighbors=min(k, len(region_df)))
            nn.fit(features)
            
            for i in range(n_to_generate):
                idx = np.random.randint(0, len(region_df))
                sample = region_df.iloc[idx]
                distances, indices = nn.kneighbors(features[idx:idx+1])
                neighbor_indices = indices[0][indices[0] != idx]
                if len(neighbor_indices) == 0:
                    neighbor_idx = idx
                else:
                    neighbor_idx = np.random.choice(neighbor_indices)
                sample_neighbor = region_df.iloc[neighbor_idx]
                gap = np.random.rand()  # interpolation factor in [0,1]
                
                synthetic_sample = {}
                synthetic_sample["batch_number"] = f"synthetic_{region}_{i+1}_{np.random.randint(10000)}"
                
                # Interpolate scalar features (explicitly convert to float).
                for col in scalar_cols:
                    try:
                        val1 = float(sample[col])
                        val2 = float(sample_neighbor[col])
                        interpolated = val1 + gap * (val2 - val1)
                    except Exception as e:
                        print(f"Error interpolating column {col}: {e}")
                        interpolated = np.nan
                    synthetic_sample[col] = interpolated
                    if np.isnan(interpolated):
                        print(f"DEBUG: {col} interpolation resulted in NaN (sample: {sample[col]}, neighbor: {sample_neighbor[col]}, gap: {gap})")
                
                # Interpolate the target t5.
                try:
                    t5_val = float(sample["t5"]) + gap * (float(sample_neighbor["t5"]) - float(sample["t5"]))
                except Exception as e:
                    print(f"Error interpolating t5: {e}")
                    t5_val = np.nan
                synthetic_sample["t5"] = t5_val
                if np.isnan(t5_val):
                    print(f"DEBUG: t5 interpolation resulted in NaN (sample: {sample['t5']}, neighbor: {sample_neighbor['t5']}, gap: {gap})")
                
                # Interpolate time-series data elementwise.
                def interpolate_ts(ts1, ts2, gap):
                    length = min(len(ts1), len(ts2))
                    syn_ts = []
                    for j in range(length):
                        try:
                            time_val = ts1[j][0] + gap * (ts2[j][0] - ts1[j][0])
                            meas_val = ts1[j][1] + gap * (ts2[j][1] - ts1[j][1])
                        except Exception as e:
                            print(f"Error interpolating time-series at index {j}: {e}")
                            time_val, meas_val = np.nan, np.nan
                        syn_ts.append([time_val, meas_val])
                    return syn_ts
                
                for col in ts_cols:
                    synthetic_sample[col] = interpolate_ts(sample[col], sample_neighbor[col], gap)
                
                # Optionally interpolate start_time and end_time if they exist.
                for time_col in ["start_time", "end_time"]:
                    if time_col in sample and pd.notnull(sample[time_col]) and pd.notnull(sample_neighbor[time_col]):
                        try:
                            t1 = sample[time_col].value
                            t2 = sample_neighbor[time_col].value
                            syn_time = pd.to_datetime(t1 + gap * (t2 - t1))
                        except Exception as e:
                            print(f"Error interpolating {time_col}: {e}")
                            syn_time = pd.NaT
                        synthetic_sample[time_col] = syn_time
                
                synthetic_rows.append(synthetic_sample)
        
        if synthetic_rows:
            synthetic_df = pd.DataFrame(synthetic_rows)
            df_final = pd.concat([df_aug.drop(columns=["region"]), synthetic_df], ignore_index=True)
        else:
            df_final = df_aug.drop(columns=["region"])
        
        return df_final
