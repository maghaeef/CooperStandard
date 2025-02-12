# %% [code]
from bounds import bounds
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# %% [markdown]
# ### 1. Read Excel Data and Organize It
# This part is essentially the same as before, parsing the Excel file and computing the ratio.

# %%
file_name = "DataOn2025Jan08.xlsx"
df1 = pd.read_excel(file_name, sheet_name="NES170K07Line2")
df2 = pd.read_excel(file_name, sheet_name="NES170K07Line1")
df = pd.concat([df1, df2], ignore_index=True)
print("Data shape:", df.shape)

# t5 thresholds from bounds
t5_lb = bounds["170K"][0]
t5_ub = bounds["170K"][1]

# %%
def safe_literal_eval(value):
    if isinstance(value, str):
        value = value.replace("nan", "None")
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None

def S1_S2_calc(S1, S2):
    S1_S2 = []
    for i in range(1, len(S1)):
        if S2[i] == 0.0:
            S1_S2.append(None)
        else:
            S1_S2.append(S1[i] / S2[i])
    return S1_S2

def organized_data(df, t5_lb, t5_ub):
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
                "time": t[1:],
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
        # Assign class based on t5 thresholds
        if row["t5"] < t5_lb:
            data[batch_number]["class"] = "low"
        elif row["t5"] > t5_ub:
            data[batch_number]["class"] = "high"
        else:
            data[batch_number]["class"] = "normal"
    
    data = {k: v for k, v in data.items() if v["MDR"] is not None and not v["MDR"].empty}
    return data

data = organized_data(df, t5_lb, t5_ub)
print(f"# low: {len({k: v for k, v in data.items() if v['class']=='low'})}")
print(f"# high: {len({k: v for k, v in data.items() if v['class']=='high'})}")
print(f"# normal: {len({k: v for k, v in data.items() if v['class']=='normal'})}")

# %% [markdown]
# ### 2. Filter by Sequence Length and Prepare Sequences
# We pad and normalize the sequences and also extract targets and the class labels.

# %%
def len_condition(data, len_threshold):
    return {k: v for k, v in data.items() if v["MDR"].shape[0] >= len_threshold}

# Filter out batches with fewer than 290 time steps
data = len_condition(data, 290)
t5_list = [v["t5"] for v in data.values()]
print("t5 range:", min(t5_list), max(t5_list))
print("Number of batches after filtering:", len(data))

# %%
def prepare_sequences(data_dict, max_len=305):
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

# Convert class labels to integers and then to one-hot encoding
# We'll use: low -> 0, normal -> 1, high -> 2
class_to_int = {'low': 0, 'normal': 1, 'high': 2}
y_class_int = np.array([class_to_int[c] for c in classes_all])
y_class = to_categorical(y_class_int, num_classes=3)

# %% [markdown]
# ### 3. Train/Test Split
# We split the data into training and testing sets while preserving class proportions.

# %%
X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
    X, y, y_class, test_size=0.2, random_state=42, stratify=y_class_int
)

# %% [markdown]
# ### 4. Multi-Task Model Definition and Training
# We build a model with a shared LSTM network and two outputs: one for t5 regression and one for classification.
# We train using a combined loss (MSE for regression and categorical crossentropy for classification).

# %%
def create_multi_task_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Masking(mask_value=0.)(inputs)
    x = LSTM(32, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(16)(x)
    x = Dropout(0.2)(x)
    shared = Dense(16, activation='relu')(x)
    
    # Regression branch for predicting t5
    regression_output = Dense(1, name='regression')(shared)
    
    # Classification branch for predicting region
    classification_output = Dense(3, activation='softmax', name='classification')(shared)
    
    model = Model(inputs=inputs, outputs=[regression_output, classification_output])
    model.compile(
        optimizer='adam',
        loss={'regression': 'mse', 'classification': 'categorical_crossentropy'},
        loss_weights={'regression': 1.0, 'classification': 1.0},
        metrics={'regression': 'mae', 'classification': 'accuracy'}
    )
    return model

model = create_multi_task_model((X_train.shape[1], X_train.shape[2]))
model.summary()

# Train the model (using more epochs to allow better learning)
history = model.fit(
    X_train, 
    {'regression': y_train, 'classification': y_class_train},
    epochs=30,
    validation_split=0.2,
    verbose=1
)

# %% [markdown]
# ### 5. Evaluation on the Test Set
# We evaluate the overall regression MAE and classification accuracy, then analyze performance per class.

# %%
# Evaluate the model on the test set
results = model.evaluate(X_test, {'regression': y_test, 'classification': y_class_test}, verbose=0)
overall_mae = results[model.metrics_names.index('regression_mae')]
overall_class_acc = results[model.metrics_names.index('classification_accuracy')]
print(f"\nOverall Test Regression MAE: {overall_mae:.4f}")
print(f"Overall Test Classification Accuracy: {overall_class_acc * 100:.2f}%")

# Get predictions
pred_reg = model.predict(X_test)[0].flatten()  # regression predictions
pred_class_probs = model.predict(X_test)[1]
pred_class_int = np.argmax(pred_class_probs, axis=1)
actual_class_int = np.argmax(y_class_test, axis=1)

# For regression, convert predicted t5 to a region using thresholds
def get_class_from_t5(value, lower_bound, upper_bound):
    if value < lower_bound:
        return 0  # low
    elif value > upper_bound:
        return 2  # high
    else:
        return 1  # normal

pred_region_from_reg = np.array([get_class_from_t5(val, t5_lb, t5_ub) for val in pred_reg])

# We'll use the classification head's predictions as the model's region decision.
# Compare these predictions to the true classes.
overall_region_accuracy = np.mean(pred_class_int == actual_class_int)
print(f"\nOverall Region Classification Accuracy (from classification head): {overall_region_accuracy * 100:.2f}%")

# Compute MAE per class (using regression predictions) and classification accuracy per class
unique_classes = [0, 1, 2]
int_to_class = {0: 'low', 1: 'normal', 2: 'high'}

for cls in unique_classes:
    indices = np.where(actual_class_int == cls)[0]
    if len(indices) == 0:
        continue
    mae_cls = np.mean(np.abs(y_test[indices] - pred_reg[indices]))
    class_acc = np.mean(pred_class_int[indices] == actual_class_int[indices])
    print(f"\nFor class '{int_to_class[cls]}':")
    print(f"  Test Regression MAE: {mae_cls:.4f}")
    print(f"  Classification Accuracy: {class_acc * 100:.2f}%")
