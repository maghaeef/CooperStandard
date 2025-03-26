# experiment_runner.py
from cooper_standard import CooperStandard
from bounds import bounds
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
import tensorflow as tf

class ExperimentRunner:
    def __init__(self, file_path, sheet_names, compound_name, variables):
        self.file_path = file_path
        self.sheet_names = sheet_names
        self.variables = variables
        self.lb, self.ub = bounds[compound_name]
        self.cooper_standard = CooperStandard(file_path, sheet_names, variables)

    def load_and_preprocess(self):
        # Load data and apply preprocessing
        df_all = self.cooper_standard.load_data()
        cleaned_df, removed_batches = self.cooper_standard.preprocessing(280)
        return cleaned_df

    def split_and_augment(self, df, use_augmentation):
        # Split the cleaned dataframe into train, validation, and test sets.
        train_df, temp_df = train_test_split(df, train_size=0.7, random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42)
        if use_augmentation:
            train_df = self.cooper_standard.balance_data_with_synthetic(train_df, self.lb, self.ub, k=5)
        return train_df, val_df, test_df

    def count_regions(self, df):
        # Count the number of datapoints for each region based on t5.
        def categorize(t5):
            if t5 < self.lb:
                return "low"
            elif t5 <= self.ub:
                return "normal"
            else:
                return "high"
        regions = df["t5"].apply(categorize)
        counts = regions.value_counts().to_dict()
        # Ensure all keys exist.
        for region in ["low", "normal", "high"]:
            if region not in counts:
                counts[region] = 0
        return counts

    def convert_and_normalize(self, train_df, val_df, fixed_length=280):
        # Always use fixed_length=280 for conversion.
        train_dict = self.cooper_standard.convert_to_dict(train_df, fixed_length=fixed_length)
        val_dict = self.cooper_standard.convert_to_dict(val_df, fixed_length=fixed_length)
        train_dict_norm = self.cooper_standard.normalize_data(train_dict, fit=True)
        val_dict_norm = self.cooper_standard.normalize_data(val_dict, fit=False)
        return train_dict_norm, val_dict_norm, fixed_length


    def prepare_data_from_dict(self, data_dict):
        # Prepare arrays for time series, scalar features, and target t5.
        time_series = np.array([batch["time_series"] for batch in data_dict.values()])
        scalar_features = np.array([batch["scalar_features"] for batch in data_dict.values()])
        targets = np.array([batch["t5"] for batch in data_dict.values()])
        return time_series, scalar_features, targets

    def build_model(self, sequence_length, num_scalar_features, lstm_units=64, scalar_dense_units=16, dense_units=32):
        # Build a dual-input neural network model with hyperparameters.
        ts_input = Input(shape=(sequence_length, 4), name='time_series_input')
        lstm_out = LSTM(units=lstm_units)(ts_input)
        scalar_input = Input(shape=(num_scalar_features,), name='scalar_input')
        scalar_dense = Dense(scalar_dense_units, activation='relu')(scalar_input)
        combined = Concatenate()([lstm_out, scalar_dense])
        dense = Dense(dense_units, activation='relu')(combined)
        output = Dense(1, name='t5_output')(dense)
        model = Model(inputs=[ts_input, scalar_input], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def grid_search_cv(self, train_df, epochs=50, batch_size=32, n_splits=4):
        # Define hyperparameter grids.
        lstm_units_grid = [8, 16, 32, 64]
        scalar_dense_units_grid = [8, 16, 32]
        dense_units_grid = [8, 16, 32]

        best_avg_mse = np.inf
        best_params = None

        # Create KFold splitter.
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        # Convert the train_df indices to a numpy array for indexing.
        train_df = train_df.reset_index(drop=True)
        X_indices = np.arange(len(train_df))

        total_combinations = len(lstm_units_grid) * len(scalar_dense_units_grid) * len(dense_units_grid)
        comb_count = 0

        for lstm_units in lstm_units_grid:
            for scalar_units in scalar_dense_units_grid:
                for dense_units in dense_units_grid:
                    comb_count += 1
                    mse_scores = []
                    # Perform CV.
                    for train_index, val_index in kf.split(X_indices):
                        cv_train_df = train_df.iloc[train_index]
                        cv_val_df = train_df.iloc[val_index]
                        # Convert and normalize data for this fold.
                        train_dict_norm, val_dict_norm, fixed_length = self.convert_and_normalize(cv_train_df, cv_val_df)
                        # Prepare data arrays.
                        train_ts, train_scalar, train_targets = self.prepare_data_from_dict(train_dict_norm)
                        val_ts, val_scalar, val_targets = self.prepare_data_from_dict(val_dict_norm)
                        sequence_length = train_ts.shape[1]
                        num_scalar_features = train_scalar.shape[1]
                        # Build and train the model with current hyperparameters.
                        model = self.build_model(sequence_length, num_scalar_features,
                                                 lstm_units=lstm_units,
                                                 scalar_dense_units=scalar_units,
                                                 dense_units=dense_units)
                        # Train model; set verbose=0 to suppress output.
                        model.fit([train_ts, train_scalar], train_targets,
                                  validation_data=([val_ts, val_scalar], val_targets),
                                  epochs=epochs, batch_size=batch_size, verbose=0)
                        # Evaluate on the fold's validation data.
                        mse, _ = model.evaluate([val_ts, val_scalar], val_targets, verbose=0)
                        mse_scores.append(mse)
                    avg_mse = np.mean(mse_scores)
                    print(f"Params: LSTM={lstm_units}, Scalar={scalar_units}, Dense={dense_units} --> Avg MSE: {avg_mse:.4f}")
                    if avg_mse < best_avg_mse:
                        best_avg_mse = avg_mse
                        best_params = {
                            "lstm_units": lstm_units,
                            "scalar_dense_units": scalar_units,
                            "dense_units": dense_units
                        }
        print(f"Best hyperparameters: {best_params} with Avg MSE: {best_avg_mse:.4f}")
        return best_params, best_avg_mse

    def compute_class_metrics(self, true_t5, pred_t5):
        # Compute MSE, MAE, and accuracy per region.
        metrics = {}
        regions = {
            "low": (true_t5 < self.lb),
            "normal": ((true_t5 >= self.lb) & (true_t5 <= self.ub)),
            "high": (true_t5 > self.ub)
        }
        for region, mask in regions.items():
            count = np.sum(mask)
            if count == 0:
                metrics[region] = {"mse": np.nan, "mae": np.nan, "accuracy": np.nan, "datapoints": 0}
            else:
                mse_val = np.mean((true_t5[mask] - pred_t5[mask]) ** 2)
                mae_val = np.mean(np.abs(true_t5[mask] - pred_t5[mask]))
                # Define region accuracy: percentage of predictions falling in the correct region.
                if region == "low":
                    acc = np.mean(pred_t5[mask] < self.lb) * 100
                elif region == "normal":
                    acc = np.mean((pred_t5[mask] >= self.lb) & (pred_t5[mask] <= self.ub)) * 100
                else:  # high
                    acc = np.mean(pred_t5[mask] > self.ub) * 100
                metrics[region] = {"mse": mse_val, "mae": mae_val, "accuracy": acc, "datapoints": int(count)}
        return metrics

    def run_experiment(self, use_augmentation, epochs=50, batch_size=32):
        # Load and preprocess data.
        df_clean = self.load_and_preprocess()
        # Split (and augment training if selected).
        train_df, val_df, test_df = self.split_and_augment(df_clean, use_augmentation)
        # Count datapoints per region.
        train_counts = self.count_regions(train_df)
        val_counts = self.count_regions(val_df)
        test_counts = self.count_regions(test_df)
        
        # --------------------
        # Hyperparameter Tuning using grid search with 4-fold CV on training data.
        print("Starting hyperparameter tuning via 4-fold CV on training set...")
        best_params, best_cv_mse = self.grid_search_cv(train_df, epochs=epochs, batch_size=batch_size, n_splits=4)
        # --------------------

        # Convert and normalize data for final training.
        train_dict_norm, val_dict_norm, fixed_length = self.convert_and_normalize(train_df, val_df)
        test_dict_norm, _, _ = self.convert_and_normalize(test_df, test_df, fixed_length=fixed_length)  # dummy for test
        # Prepare arrays.
        train_ts, train_scalar, train_targets = self.prepare_data_from_dict(train_dict_norm)
        val_ts, val_scalar, val_targets = self.prepare_data_from_dict(val_dict_norm)
        test_ts, test_scalar, test_targets = self.prepare_data_from_dict(test_dict_norm)
        sequence_length = train_ts.shape[1]
        num_scalar_features = train_scalar.shape[1]
        # Build final model with best hyperparameters.
        final_model = self.build_model(sequence_length, num_scalar_features,
                                       lstm_units=best_params["lstm_units"],
                                       scalar_dense_units=best_params["scalar_dense_units"],
                                       dense_units=best_params["dense_units"])
        # Train final model.
        final_model.fit([train_ts, train_scalar], train_targets,
                        validation_data=([val_ts, val_scalar], val_targets),
                        epochs=epochs, batch_size=batch_size, verbose=1)
        # Evaluate overall performance.
        train_loss, train_mae = final_model.evaluate([train_ts, train_scalar], train_targets, verbose=0)
        val_loss, val_mae = final_model.evaluate([val_ts, val_scalar], val_targets, verbose=0)
        test_loss, test_mae = final_model.evaluate([test_ts, test_scalar], test_targets, verbose=0)
        # Make predictions.
        train_pred = final_model.predict([train_ts, train_scalar]).flatten()
        val_pred = final_model.predict([val_ts, val_scalar]).flatten()
        test_pred = final_model.predict([test_ts, test_scalar]).flatten()
        # Compute per-region metrics.
        train_metrics = self.compute_class_metrics(train_targets, train_pred)
        val_metrics = self.compute_class_metrics(val_targets, val_pred)
        test_metrics = self.compute_class_metrics(test_targets, test_pred)
        # Bundle the results.
        results = {
            "augmentation": use_augmentation,
            "best_hyperparameters": best_params,
            "best_cv_mse": best_cv_mse,
            "datapoints": {
                "train": train_counts,
                "val": val_counts,
                "test": test_counts
            },
            "performance": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "overall": {
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "test_loss": test_loss,
                    "test_mae": test_mae
                }
            }
        }
        return results