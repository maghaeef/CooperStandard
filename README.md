# CooperStandard Project

The **CooperStandard** project provides a comprehensive pipeline for processing batch process data and training a neural network model for predicting the `t5` performance metric. The project includes modules for data loading, preprocessing (including handling of time-series data), normalization, and optional data augmentation using a SMOTE-like synthetic sampling technique. It then uses a **dual-input neural network** architecture *(combining LSTM for time-series and Dense layers for scalar features)* to perform regression.

## Repository Structure

- **cooper_standard.py**  
  Contains the `CooperStandard` class that:
  - Loads data from an Excel file (across multiple sheets).
  - Preprocesses the data (truncating, interpolating, and computing derived features).
  - Converts the data into a dictionary format for further processing.
  - Normalizes the data using min-max scaling.
  - Splits the data into training, validation, and test sets.
  - Provides a SMOTE-like method for synthetic data generation to balance the dataset.

- **experiment_runner.py**  
  Defines the `ExperimentRunner` class that encapsulates the experiment workflow:
  - Loads and preprocesses the data.
  - Splits the data and (optionally) augments the training set.
  - Converts and normalizes the data.
  - Builds and trains a neural network model.
  - Evaluates the model overall as well as on a per-region basis (low, normal, high).
  - Collects performance metrics (MSE, MAE, and accuracy) and data point counts per region.

- **main_experiment.py**  
  The main script to run experiments under two scenarios:
  1. Without data augmentation (`USE_AUGMENTATION=False`).
  2. With data augmentation (`USE_AUGMENTATION=True`).

  It then aggregates the results into a pandas DataFrame and saves them to a CSV file.

- **bounds.py** (or similar)  
  Contains the threshold definitions used to categorize the `t5` values (e.g., lower and upper bounds).

## Setup and Requirements

- **Python 3.x**
- **Packages:**
  - pandas
  - numpy
  - scikit-learn
  - keras (and tensorflow as backend)
  - matplotlib
  - openpyxl (for reading Excel files)

Install the necessary packages using pip:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib openpyxl
