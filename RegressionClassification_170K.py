# %% [code]
from bounds import bounds
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# %% [markdown]
# ### 1. Read Excel Data and Organize It

# %%
file_name = "DataOn2025Jan08.xlsx"
df1 = pd.read_excel(file_name, sheet_name="NES170K07Line2")
df2 = pd.read_excel(file_name, sheet_name="NES170K07Line1")
df = pd.concat([df1, df2], ignore_index=True)
print("Data shape:", df.shape)

# Get t5 thresholds from bounds dictionary
t5_lb = bounds["170K"][0]
t5_ub = bounds["170K"][1]

# %%
def safe_literal_eval(value):
    """Safely evaluate a string representation, replacing 'nan' with None."""
    if isinstance(value, str):
        value = value.replace("nan", "None")
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None

def S1_S2_calc(S1, S2):
    """Calculate the element-wise ratio S1/S2 (skip first element, avoid division by zero)."""
    S1_S2 = []
    for i in range(1, len(S1)):
        if S2[i] == 0.0:
            S1_S2.append(None)
        else:
            S1_S2.append(S1[i] / S2[i])
    return S1_S2

def organized_data(df, t5_lb, t5_ub):
    """
    Process each row to extract the time-series (MDR), target t5,
    and assign a region label ('low', 'normal', 'high') based on thresholds.
    """
    data = {}
    for index, row in df.iterrows():
        if pd.isna(row['t5']):
            continue
        batch_number = row["batch_number"]
        data[batch_number] = {"MDR": None, "t5": row["t5"], "class": None}
        
        t_S1 = safe_literal_eval(row["MDRTorqueS1"])
        t_S2 = safe_literal_eval(row["MDRTorqueS2"])
        if t_S1 is not None and t_S2 is not None:
            t, S1 = zip(*t_S1)
            t, S2 = zip(*t_S2)
            t, S1, S2 = list(t), list(S1), list(S2)
            S1_S2 = S1_S2_calc(S1, S2)
            MDR = pd.DataFrame({
                "time": t[1:],    # Exclude first element
                "S1": S1[1:],
                "S2": S2[1:],
                "S1_S2": S1_S2
            })
            MDR.interpolate(method="linear", inplace=True, limit_direction="both")
            MDR.fillna(method="bfill", inplace=True)
            MDR.fillna(method="ffill", inplace=True)
        else:
            continue
        
        data[batch_number]["MDR"] = MDR
        # Assign class label based on t5 thresholds (for evaluation purposes)
        if row["t5"] < t5_lb:
            data[batch_number]["class"] = "low"
        elif row["t5"] > t5_ub:
            data[batch_number]["class"] = "high"
        else:
            data[batch_number]["class"] = "normal"
    
    # Remove batches with empty MDR
    data = {k: v for k, v in data.items() if v["MDR"] is not None and not v["MDR"].empty}
    return data

data = organized_data(df, t5_lb, t5_ub)
print(f"# low: {len({k: v for k, v in data.items() if v['class']=='low'})}")
print(f"# high: {len({k: v for k, v in data.items() if v['class']=='high'})}")
print(f"# normal: {len({k: v for k, v in data.items() if v['class']=='normal'})}")

# %% [markdown]
# ### 2. Filter by Sequence Length and Prepare Sequences

# %%
# def len_condition(data, len_threshold):
#     """Keep batches with at least len_threshold time steps."""
#     return {k: v for k, v in data.items() if v["MDR"].shape[0] >= len_threshold}

# # Filter out batches with fewer than 290 time steps
# data = len_condition(data, 290)



t5_list = [v["t5"] for v in data.values()]
print("t5 range:", min(t5_list), max(t5_list))
print("Number of batches after filtering:", len(data))

# %%
def prepare_sequences(data_dict, max_len=305):
    """
    Process the data into padded sequences, targets (t5), and class labels.
    Returns:
      X: Padded & normalized sequences (n_samples, max_len, 3)
      y: Array of t5 values
      classes: Array of class labels ('low', 'normal', 'high')
      scalers: List of StandardScalers (one per feature)
    """
    sequences = []
    targets = []
    classes = []
    for _id in data_dict:
        df_batch = data_dict[_id]["MDR"]
        seq = df_batch[['S1', 'S2', 'S1_S2']].values.astype('float32')
        sequences.append(seq)
        targets.append(data_dict[_id]["t5"])
        classes.append(data_dict[_id]["class"])
    
    non_empty = [s for s in sequences if len(s) > 0]
    filtered_targets = [t for s, t in zip(sequences, targets) if len(s) > 0]
    filtered_classes = [c for s, c in zip(sequences, classes) if len(s) > 0]
    
    padded_sequences = pad_sequences(
        non_empty,
        maxlen=max_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )
    
    scalers = []
    normalized = []
    for feature_idx in range(padded_sequences.shape[2]):
        feature_data = padded_sequences[:, :, feature_idx].reshape(-1, 1)
        scaler = StandardScaler().fit(feature_data)
        scalers.append(scaler)
        normalized_feature = scaler.transform(feature_data).reshape(
            padded_sequences.shape[0], padded_sequences.shape[1], 1)
        normalized.append(normalized_feature)
    
    X = np.concatenate(normalized, axis=2)
    return X, np.array(filtered_targets), np.array(filtered_classes), scalers

X, y, classes_all, scalers = prepare_sequences(data)

# %% [markdown]
# ### 3. Train/Test Split
# We use stratified splitting to maintain class proportions for evaluation.

# %%
X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split(
    X, y, classes_all, test_size=0.2, random_state=42, stratify=classes_all
)

# %% [markdown]
# ### 4. Define a Custom Loss Function with a Region Penalty
# The loss adds a penalty when y_pred falls into a different region than y_true.

# %%
def custom_loss(t5_lb, t5_ub, penalty_weight=5.0):
    """
    Returns a loss function that computes MSE plus a penalty if the predicted t5
    is in a different region than the true t5.
    """
    def loss(y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)  # shape (batch_size,)
        # Flatten true and predicted values
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        # Compute regions: 0 = low, 1 = normal, 2 = high.
        region_true = tf.where(y_true_flat < t5_lb, 0.0, tf.where(y_true_flat > t5_ub, 2.0, 1.0))
        region_pred = tf.where(y_pred_flat < t5_lb, 0.0, tf.where(y_pred_flat > t5_ub, 2.0, 1.0))
        # Penalty is 1 if regions differ, 0 if same.
        region_penalty = tf.cast(tf.not_equal(region_true, region_pred), tf.float32)
        # Combine MSE and penalty; note mse is computed per sample.
        return tf.reduce_mean(mse + penalty_weight * region_penalty)
    return loss

# %% [markdown]
# ### 5. Build and Train the Regression Model (No Classification Head)
# We now create a model and compile it with the custom loss.

# %%
def create_lstm_model_reg(input_shape):
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        LSTM(16),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model

model = create_lstm_model_reg((X_train.shape[1], X_train.shape[2]))
loss_fn = custom_loss(t5_lb, t5_ub, penalty_weight=5.0)
model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
model.summary()

# Train the model (increase epochs as needed)
history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_split=0.2,
    verbose=1
)

# %% [markdown]
# ### 6. Evaluation: Regression MAE and Region Accuracy
# We compute the overall MAE and then check, per class, whether the predicted t5 falls in the correct region.

# %%
# Evaluate overall test performance
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nOverall Test Regression MAE: {test_mae:.4f}")

# Get regression predictions
predictions = model.predict(X_test).flatten()

# Define a helper function to assign region based on t5 thresholds
def get_region(value, lower_bound, upper_bound):
    if value < lower_bound:
        return "low"
    elif value > upper_bound:
        return "high"
    else:
        return "normal"

# Compute predicted regions from predicted t5 values and actual regions from true t5 values
predicted_regions = np.array([get_region(val, t5_lb, t5_ub) for val in predictions])
actual_regions = np.array(classes_test)

# Overall region accuracy: percent of predictions in the correct region
overall_region_accuracy = np.mean(predicted_regions == actual_regions)
print(f"Overall Region Accuracy: {overall_region_accuracy * 100:.2f}%")

# Compute MAE and region accuracy per class
unique_regions = np.unique(actual_regions)
for region in unique_regions:
    indices = np.where(actual_regions == region)[0]
    mae_region = np.mean(np.abs(y_test[indices] - predictions[indices]))
    region_acc = np.mean(predicted_regions[indices] == actual_regions[indices])
    print(f"\nFor class '{region}':")
    print(f"  Test MAE: {mae_region:.4f}")
    print(f"  Region Accuracy: {region_acc * 100:.2f}%")
