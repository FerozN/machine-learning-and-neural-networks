# -*- coding: utf-8 -*-
"""Neural Networks: Polynomial Regression and XOR

Keras feedforward networks fitting y = 0.2x^4 + 2x^3 + 0.1x^2 + 10 across
architectures, activations, shuffling, and scaling, plus a 2-2-1 network that
learns XOR with backpropagation written from scratch in NumPy.

Author: Feroz Naeem
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate 30,000 samples for x in the interval [-1, 1]
x = np.linspace(-1, 1, 30000)
# Compute y based on the given function
y = 0.2 * x**4 + 2 * x**3 + 0.1 * x**2 + 10

# Plot the function
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = 0.2*x^4 + 2*x^3 + 0.1*x^2 + 10')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def create_xy_data(num_samples=30000, start=-1, end=1):
    """
    Creates x and y data for the function y = 0.2*x^4 + 2*x^3 + 0.1*x^2 + 10.
    Also plots the function as required for Q1.

    Parameters:
    -----------
    num_samples : int
        Number of data points to generate.
    start : float
        Starting value of the range for x.
    end : float
        Ending value of the range for x.

    Returns:
    --------
    x : np.ndarray
        Generated x values of shape (num_samples,).
    y : np.ndarray
        Generated y values of shape (num_samples,).
    """
    # Generate x values in [start, end]
    x = np.linspace(start, end, num_samples)

    # Compute y using the given function
    y = 0.2 * x**4 + 2 * x**3 + 0.1 * x**2 + 10

    # Plot the function
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='y = 0.2*x^4 + 2*x^3 + 0.1*x^2 + 10')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Initial Plot of the Function')
    plt.legend()
    plt.show()

    return x, y


def get_dataset(shuffle_data=False, num_samples=30000):
    """
    Generates x, y data for the function y = 0.2*x^4 + 2*x^3 + 0.1*x^2 + 10.
    Optionally shuffles the data based on the 'shuffle_data' argument.

    Parameters:
    -----------
    shuffle_data : bool
        If True, shuffle the (x, y) pairs before returning.
    num_samples : int
        Number of data points to generate.

    Returns:
    --------
    x : np.ndarray
        Generated (and possibly shuffled) x values of shape (num_samples,).
    y : np.ndarray
        Generated (and possibly shuffled) y values of shape (num_samples,).
    """
    x = np.linspace(-1, 1, num_samples)
    y = 0.2 * x**4 + 2 * x**3 + 0.1 * x**2 + 10

    if shuffle_data:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

    return x, y


# Example usage (optional):
if __name__ == "__main__":
    # For Q1: Create and plot data
    x_data, y_data = create_xy_data()  # This will also display the plot

    # For the shuffling requirement
    x_shuffled, y_shuffled = get_dataset(shuffle_data=True)
    print("First 5 x values (shuffled):", x_shuffled[:5])
    print("First 5 y values (shuffled):", y_shuffled[:5])

def split_data(x, y, train_ratio, val_ratio, test_ratio):
    """
    Splits the data into training, validation, and test sets based on provided ratios.

    Parameters:
    -----------
    x : np.ndarray
        Array of input data.
    y : np.ndarray
        Array of corresponding output data.
    train_ratio : float
        Proportion of data to use for training (e.g., 0.3 for 30%).
    val_ratio : float
        Proportion of data to use for validation (e.g., 0.2 for 20%).
    test_ratio : float
        Proportion of data to use for testing (e.g., 0.5 for 50%).

    Returns:
    --------
    x_train, y_train, x_val, y_val, x_test, y_test : tuple of np.ndarrays
        The split data arrays.
    """
    # Optional: Normalize ratios if they do not sum to 1 exactly
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    n_samples = len(x)
    # Calculate the indices where to split the data
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    # Split the arrays
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test

# --- Testing the split_data function ---
# Assuming you already generated or loaded your data using get_dataset
# For example, use shuffled data from your get_dataset function:
x_data, y_data = get_dataset(shuffle_data=True)

# Now, split the data with the specified ratios: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# Print out the sizes of each split to verify
print("Training set size:", len(x_train))
print("Validation set size:", len(x_val))
print("Test set size:", len(x_test))

def scale_data(x, y):
    """
    Scales x and y to the range [0, 1] using min-max normalization.

    Parameters:
    -----------
    x : np.ndarray
        Input feature array (e.g., x values).
    y : np.ndarray
        Target array (e.g., y values).

    Returns:
    --------
    x_scaled : np.ndarray
        x scaled to [0, 1].
    y_scaled : np.ndarray
        y scaled to [0, 1].
    """
    # Scale x
    x_min, x_max = x.min(), x.max()
    x_scaled = (x - x_min) / (x_max - x_min)

    # Scale y
    y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min)

    return x_scaled, y_scaled

# Example usage:
if __name__ == "__main__":
    # Let's assume we already have x_data, y_data from get_dataset(...)
    x_data, y_data = get_dataset(shuffle_data=True)

    # Scale them
    x_scaled, y_scaled = scale_data(x_data, y_data)

    print("Original x range:", x_data.min(), "to", x_data.max())
    print("Scaled x range:  ", x_scaled.min(), "to", x_scaled.max())

    print("Original y range:", y_data.min(), "to", y_data.max())
    print("Scaled y range:  ", y_scaled.min(), "to", y_scaled.max())

def calculate_metrics(y_true, y_pred):
    """
    Calculates mean absolute error (MAE), mean square error (MSE),
    root mean square error (RMSE), and R^2 score given actual and
    predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        Array of actual (target) values.
    y_pred : np.ndarray
        Array of predicted values.

    Returns:
    --------
    mae : float
        Mean Absolute Error
    mse : float
        Mean Square Error
    rmse : float
        Root Mean Square Error
    r2 : float
        R^2 Score
    """
    # MAE: average of absolute differences
    mae = np.mean(np.abs(y_true - y_pred))

    # MSE: average of squared differences
    mse = np.mean((y_true - y_pred)**2)

    # RMSE: square root of MSE
    rmse = np.sqrt(mse)

    # R^2: coefficient of determination
    # Formula: R^2 = 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)

    return mae, mse, rmse, r2

"""#Case 1"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 1: Shuffled and Unscaled Data, Structure 1, ReLU activation ---

# 1. Get data (shuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=True)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# Reshape x arrays to be 2D (Keras expects inputs in shape (n_samples, n_features))
x_train = x_train.reshape(-1, 1)
x_val   = x_val.reshape(-1, 1)
x_test  = x_test.reshape(-1, 1)

# 3. Build the model for Structure 1:
#    - First hidden layer: 12 units with ReLU activation
#    - Second hidden layer: 8 units with ReLU activation
#    - Third hidden layer: 4 units with ReLU activation
#    - Output layer: 1 unit (for regression; linear activation by default)
model = Sequential([
    Dense(12, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)  # Output layer for regression
])

# 4. Compile the model using MSE as the loss and Adam as the optimizer.
model.compile(optimizer='adam', loss='mse')

# 5. Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=12,
                    validation_data=(x_val, y_val), verbose=1)

# 6. Predict on the test set
y_pred = model.predict(x_test).flatten()  # flatten to convert from 2D to 1D array

# 7. Calculate evaluation metrics using your previously defined function
mae, mse_val, rmse, r2 = calculate_metrics(y_test, y_pred)
print("Metrics on Test Set:")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 8. Plot the results: Actual vs. Predicted
plt.figure(figsize=(12, 5))

# Graph 1: Actual Test Data
plt.subplot(1, 2, 1)
plt.scatter(x_test, y_test, color='blue', s=10, label='Actual')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Actual Test Data')
plt.legend()

# Graph 2: Predicted Data
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_pred, color='red', s=10, label='Predicted')
plt.xlabel('x_test')
plt.ylabel('y_pred')
plt.title('Predicted Test Data')
plt.legend()

plt.tight_layout()
plt.show()

"""#Case 1 unshuffled

"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 1: Unshuffled and Unscaled Data, Structure 1, ReLU activation ---

# 1. Get data (unshuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=False)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# Reshape x arrays to be 2D (Keras expects inputs in shape (n_samples, n_features))
x_train = x_train.reshape(-1, 1)
x_val   = x_val.reshape(-1, 1)
x_test  = x_test.reshape(-1, 1)

# 3. Build the model for Structure 1:
#    - First hidden layer: 12 units with ReLU activation
#    - Second hidden layer: 8 units with ReLU activation
#    - Third hidden layer: 4 units with ReLU activation
#    - Output layer: 1 unit (for regression; linear activation by default)
model = Sequential([
    Dense(12, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)  # Output layer for regression
])

# 4. Compile the model using MSE as the loss and Adam as the optimizer.
model.compile(optimizer='adam', loss='mse')

# 5. Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=12,
                    validation_data=(x_val, y_val), verbose=1)

# 6. Predict on the test set
y_pred = model.predict(x_test).flatten()  # flatten to convert from 2D to 1D array

# 7. Calculate evaluation metrics using the previously defined function
mae, mse_val, rmse, r2 = calculate_metrics(y_test, y_pred)
print("Metrics on Test Set (Unshuffled Data):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 8. Plot the results: Actual vs. Predicted
plt.figure(figsize=(12, 5))

# Graph 1: Actual Test Data
plt.subplot(1, 2, 1)
plt.scatter(x_test, y_test, color='blue', s=10, label='Actual')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Actual Test Data (Unshuffled)')
plt.legend()

# Graph 2: Predicted Data
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_pred, color='red', s=10, label='Predicted')
plt.xlabel('x_test')
plt.ylabel('y_pred')
plt.title('Predicted Test Data (Unshuffled)')
plt.legend()

plt.tight_layout()
plt.show()

"""#Case 2"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 2: Shuffled and Unscaled Data, Structure 2, ReLU activation ---

# 1. Get data (shuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=True)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# Reshape x arrays to be 2D (required for Keras: shape=(n_samples, n_features))
x_train = x_train.reshape(-1, 1)
x_val   = x_val.reshape(-1, 1)
x_test  = x_test.reshape(-1, 1)

# 3. Build the model for Structure 2:
#    - One hidden layer: 24 units with ReLU activation.
#    - Output layer: 1 unit (for regression).
model = Sequential([
    Dense(24, activation='relu', input_shape=(1,)),
    Dense(1)  # Output layer: linear activation (default) for regression.
])

# 4. Compile the model using the Adam optimizer and MSE as the loss function.
model.compile(optimizer='adam', loss='mse')

# 5. Train the model for 20 epochs with a batch size of 12.
history = model.fit(x_train, y_train, epochs=20, batch_size=12,
                    validation_data=(x_val, y_val), verbose=1)

# 6. Predict on the test set.
y_pred = model.predict(x_test).flatten()  # Flatten to convert to 1D array

# 7. Calculate evaluation metrics.
mae, mse_val, rmse, r2 = calculate_metrics(y_test, y_pred)
print("Metrics on Test Set (Case 2, Shuffled Data):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 8. Plot Actual vs. Predicted data.
plt.figure(figsize=(12, 5))

# Graph 1: Actual Test Data.
plt.subplot(1, 2, 1)
plt.scatter(x_test, y_test, color='blue', s=10, label='Actual')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Actual Test Data (Shuffled)')
plt.legend()

# Graph 2: Predicted Test Data.
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_pred, color='red', s=10, label='Predicted')
plt.xlabel('x_test')
plt.ylabel('y_pred')
plt.title('Predicted Test Data (Shuffled)')
plt.legend()

plt.tight_layout()
plt.show()

"""#Case 2 unshuffled"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 2: Unshuffled and Unscaled Data, Structure 2, ReLU activation ---

# 1. Get data (unshuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=False)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# Reshape x arrays to be 2D
x_train = x_train.reshape(-1, 1)
x_val   = x_val.reshape(-1, 1)
x_test  = x_test.reshape(-1, 1)

# 3. Build the model for Structure 2:
#    - One hidden layer: 24 units with ReLU activation.
#    - Output layer: 1 unit for regression.
model = Sequential([
    Dense(24, activation='relu', input_shape=(1,)),
    Dense(1)
])

# 4. Compile the model using the Adam optimizer and MSE as the loss function.
model.compile(optimizer='adam', loss='mse')

# 5. Train the model for 20 epochs with a batch size of 12.
history = model.fit(x_train, y_train, epochs=20, batch_size=12,
                    validation_data=(x_val, y_val), verbose=1)

# 6. Predict on the test set.
y_pred = model.predict(x_test).flatten()

# 7. Calculate evaluation metrics.
mae, mse_val, rmse, r2 = calculate_metrics(y_test, y_pred)
print("Metrics on Test Set (Case 2, Unshuffled Data):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 8. Plot Actual vs. Predicted data.
plt.figure(figsize=(12, 5))

# Graph 1: Actual Test Data.
plt.subplot(1, 2, 1)
plt.scatter(x_test, y_test, color='blue', s=10, label='Actual')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Actual Test Data (Unshuffled)')
plt.legend()

# Graph 2: Predicted Test Data.
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_pred, color='red', s=10, label='Predicted')
plt.xlabel('x_test')
plt.ylabel('y_pred')
plt.title('Predicted Test Data (Unshuffled)')
plt.legend()

plt.tight_layout()
plt.show()

"""#Case 3"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 3: Shuffled and Unscaled Data, Structure 1, Activation: tanh ---

# 1. Get data (shuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=True)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# Reshape x arrays to be 2D (Keras requires shape (n_samples, n_features))
x_train = x_train.reshape(-1, 1)
x_val   = x_val.reshape(-1, 1)
x_test  = x_test.reshape(-1, 1)

# 3. Build the model for Structure 1 with tanh activation:
#    - First hidden layer: 12 units with tanh activation
#    - Second hidden layer: 8 units with tanh activation
#    - Third hidden layer: 4 units with tanh activation
#    - Output layer: 1 unit (for regression; linear activation by default)
model = Sequential([
    Dense(12, activation='tanh', input_shape=(1,)),
    Dense(8, activation='tanh'),
    Dense(4, activation='tanh'),
    Dense(1)  # Output layer for regression
])

# 4. Compile the model using Adam as the optimizer and MSE as the loss function.
model.compile(optimizer='adam', loss='mse')

# 5. Train the model for 20 epochs with a batch size of 12.
history = model.fit(x_train, y_train, epochs=20, batch_size=12,
                    validation_data=(x_val, y_val), verbose=1)

# 6. Predict on the test set.
y_pred = model.predict(x_test).flatten()  # Flatten to convert from 2D to 1D array

# 7. Calculate evaluation metrics (MAE, MSE, RMSE, and R²).
mae, mse_val, rmse, r2 = calculate_metrics(y_test, y_pred)
print("Metrics on Test Set (Case 3):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 8. Plot the results: Actual vs. Predicted
plt.figure(figsize=(12, 5))

# Graph 1: Actual Test Data
plt.subplot(1, 2, 1)
plt.scatter(x_test, y_test, color='blue', s=10, label='Actual')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Actual Test Data (Case 3: tanh)')
plt.legend()

# Graph 2: Predicted Data
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_pred, color='red', s=10, label='Predicted')
plt.xlabel('x_test')
plt.ylabel('y_pred')
plt.title('Predicted Test Data (Case 3: tanh)')
plt.legend()

plt.tight_layout()
plt.show()

"""#Case 3 unshuffled"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 3: Shuffled and Unscaled Data, Structure 1, Activation: tanh ---

# 1. Get data (shuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=False)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# Reshape x arrays to be 2D (Keras requires shape (n_samples, n_features))
x_train = x_train.reshape(-1, 1)
x_val   = x_val.reshape(-1, 1)
x_test  = x_test.reshape(-1, 1)

# 3. Build the model for Structure 1 with tanh activation:
#    - First hidden layer: 12 units with tanh activation
#    - Second hidden layer: 8 units with tanh activation
#    - Third hidden layer: 4 units with tanh activation
#    - Output layer: 1 unit (for regression; linear activation by default)
model = Sequential([
    Dense(12, activation='tanh', input_shape=(1,)),
    Dense(8, activation='tanh'),
    Dense(4, activation='tanh'),
    Dense(1)  # Output layer for regression
])

# 4. Compile the model using Adam as the optimizer and MSE as the loss function.
model.compile(optimizer='adam', loss='mse')

# 5. Train the model for 20 epochs with a batch size of 12.
history = model.fit(x_train, y_train, epochs=20, batch_size=12,
                    validation_data=(x_val, y_val), verbose=1)

# 6. Predict on the test set.
y_pred = model.predict(x_test).flatten()  # Flatten to convert from 2D to 1D array

# 7. Calculate evaluation metrics (MAE, MSE, RMSE, and R²).
mae, mse_val, rmse, r2 = calculate_metrics(y_test, y_pred)
print("Metrics on Test Set Unshuffled(Case 3):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 8. Plot the results: Actual vs. Predicted
plt.figure(figsize=(12, 5))

# Graph 1: Actual Test Data
plt.subplot(1, 2, 1)
plt.scatter(x_test, y_test, color='blue', s=10, label='Actual')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Actual Test Data (Case 3: tanh)')
plt.legend()

# Graph 2: Predicted Data
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_pred, color='red', s=10, label='Predicted')
plt.xlabel('x_test')
plt.ylabel('y_pred')
plt.title('Predicted Test Data (Case 3: tanh)')
plt.legend()

plt.tight_layout()
plt.show()

"""#Case 4"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 4: Shuffled and Scaled Data, Structure 1, Activation: ReLU ---

# 1. Get data (shuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=True)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# 3. Scale the data (both x and y) for each set.
x_train_scaled, y_train_scaled = scale_data(x_train, y_train)
x_val_scaled,   y_val_scaled   = scale_data(x_val, y_val)
x_test_scaled,  y_test_scaled  = scale_data(x_test, y_test)

# 4. Reshape the x arrays to 2D for Keras (shape: (n_samples, 1))
x_train_scaled = x_train_scaled.reshape(-1, 1)
x_val_scaled   = x_val_scaled.reshape(-1, 1)
x_test_scaled  = x_test_scaled.reshape(-1, 1)

# 5. Build the model for Structure 1 using ReLU activation:
#    - First hidden layer: 12 units with ReLU
#    - Second hidden layer: 8 units with ReLU
#    - Third hidden layer: 4 units with ReLU
#    - Output layer: 1 unit for regression (linear activation by default)
model = Sequential([
    Dense(12, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])

# 6. Compile the model with Adam optimizer and MSE loss.
model.compile(optimizer='adam', loss='mse')

# 7. Train the model for 20 epochs with a batch size of 12.
history = model.fit(x_train_scaled, y_train_scaled, epochs=20, batch_size=12,
                    validation_data=(x_val_scaled, y_val_scaled), verbose=1)

# 8. Predict on the test set.
y_pred_scaled = model.predict(x_test_scaled).flatten()

# 9. Calculate evaluation metrics using the scaled test data.
mae, mse_val, rmse, r2 = calculate_metrics(y_test_scaled, y_pred_scaled)
print("Metrics on Scaled Test Set (Case 4):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 10. Plot the results: Actual vs. Predicted on the scaled test data.
plt.figure(figsize=(12, 5))

# Graph 1: Actual Test Data (scaled)
plt.subplot(1, 2, 1)
plt.scatter(x_test_scaled, y_test_scaled, color='blue', s=10, label='Actual')
plt.xlabel('x_test (scaled)')
plt.ylabel('y_test (scaled)')
plt.title('Actual Scaled Test Data')
plt.legend()

# Graph 2: Predicted Test Data (scaled)
plt.subplot(1, 2, 2)
plt.scatter(x_test_scaled, y_pred_scaled, color='red', s=10, label='Predicted')
plt.xlabel('x_test (scaled)')
plt.ylabel('y_pred (scaled)')
plt.title('Predicted Scaled Test Data')
plt.legend()

plt.tight_layout()
plt.show()

"""#Case 4"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 4 (Unshuffled): Unshuffled and Scaled Data, Structure 1, Activation: ReLU ---

# 1. Get data (unshuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=False)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# 3. Scale the data (both x and y) for each set.
x_train_scaled, y_train_scaled = scale_data(x_train, y_train)
x_val_scaled,   y_val_scaled   = scale_data(x_val, y_val)
x_test_scaled,  y_test_scaled  = scale_data(x_test, y_test)

# 4. Reshape the x arrays to 2D (Keras requires shape (n_samples, n_features))
x_train_scaled = x_train_scaled.reshape(-1, 1)
x_val_scaled   = x_val_scaled.reshape(-1, 1)
x_test_scaled  = x_test_scaled.reshape(-1, 1)

# 5. Build the model for Structure 1 using ReLU activation:
#    - First hidden layer: 12 units with ReLU
#    - Second hidden layer: 8 units with ReLU
#    - Third hidden layer: 4 units with ReLU
#    - Output layer: 1 unit for regression (linear activation by default)
model = Sequential([
    Dense(12, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])

# 6. Compile the model with Adam optimizer and MSE loss.
model.compile(optimizer='adam', loss='mse')

# 7. Train the model for 20 epochs with a batch size of 12.
history = model.fit(x_train_scaled, y_train_scaled, epochs=20, batch_size=12,
                    validation_data=(x_val_scaled, y_val_scaled), verbose=1)

# 8. Predict on the test set.
y_pred_scaled = model.predict(x_test_scaled).flatten()

# 9. Calculate evaluation metrics using the scaled test data.
mae, mse_val, rmse, r2 = calculate_metrics(y_test_scaled, y_pred_scaled)
print("Metrics on Scaled Test Set (Case 4, Unshuffled Data):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 10. Plot the results: Actual vs. Predicted on the scaled test data.
plt.figure(figsize=(12, 5))

# Graph 1: Actual Test Data (scaled)
plt.subplot(1, 2, 1)
plt.scatter(x_test_scaled, y_test_scaled, color='blue', s=10, label='Actual')
plt.xlabel('x_test (scaled)')
plt.ylabel('y_test (scaled)')
plt.title('Actual Scaled Test Data (Unshuffled)')
plt.legend()

# Graph 2: Predicted Test Data (scaled)
plt.subplot(1, 2, 2)
plt.scatter(x_test_scaled, y_pred_scaled, color='red', s=10, label='Predicted')
plt.xlabel('x_test (scaled)')
plt.ylabel('y_pred (scaled)')
plt.title('Predicted Scaled Test Data (Unshuffled)')
plt.legend()

plt.tight_layout()
plt.show()

"""#Case 5"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 5: Shuffled and Scaled Data, Structure 1, Activation: tanh ---

# 1. Get data (shuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=True)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# 3. Scale the data (both x and y) for each set.
x_train_scaled, y_train_scaled = scale_data(x_train, y_train)
x_val_scaled,   y_val_scaled   = scale_data(x_val, y_val)
x_test_scaled,  y_test_scaled  = scale_data(x_test, y_test)

# 4. Reshape the x arrays to 2D (Keras expects shape (n_samples, n_features))
x_train_scaled = x_train_scaled.reshape(-1, 1)
x_val_scaled   = x_val_scaled.reshape(-1, 1)
x_test_scaled  = x_test_scaled.reshape(-1, 1)

# 5. Build the model for Structure 1 using tanh activation:
#    - First hidden layer: 12 units with tanh activation
#    - Second hidden layer: 8 units with tanh activation
#    - Third hidden layer: 4 units with tanh activation
#    - Output layer: 1 unit (regression; linear activation by default)
model = Sequential([
    Dense(12, activation='tanh', input_shape=(1,)),
    Dense(8, activation='tanh'),
    Dense(4, activation='tanh'),
    Dense(1)
])

# 6. Compile the model using Adam optimizer and MSE loss.
model.compile(optimizer='adam', loss='mse')

# 7. Train the model for 20 epochs with a batch size of 12.
history = model.fit(x_train_scaled, y_train_scaled, epochs=20, batch_size=12,
                    validation_data=(x_val_scaled, y_val_scaled), verbose=1)

# 8. Predict on the test set.
y_pred_scaled = model.predict(x_test_scaled).flatten()

# 9. Calculate evaluation metrics on the scaled test data.
mae, mse_val, rmse, r2 = calculate_metrics(y_test_scaled, y_pred_scaled)
print("Metrics on Scaled Test Set (Case 5, Shuffled Data):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 10. Plot Actual vs. Predicted on the scaled test data.
plt.figure(figsize=(12, 5))

# Graph 1: Actual Scaled Test Data.
plt.subplot(1, 2, 1)
plt.scatter(x_test_scaled, y_test_scaled, color='blue', s=10, label='Actual')
plt.xlabel('x_test (scaled)')
plt.ylabel('y_test (scaled)')
plt.title('Actual Scaled Test Data (Shuffled)')
plt.legend()

# Graph 2: Predicted Scaled Test Data.
plt.subplot(1, 2, 2)
plt.scatter(x_test_scaled, y_pred_scaled, color='red', s=10, label='Predicted')
plt.xlabel('x_test (scaled)')
plt.ylabel('y_pred (scaled)')
plt.title('Predicted Scaled Test Data (Shuffled)')
plt.legend()

plt.tight_layout()
plt.show()

"""Case 5 unshuffled"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- CASE 5: Unshuffled and Scaled Data, Structure 1, Activation: tanh ---

# 1. Get data (unshuffled and unscaled)
x_data, y_data = get_dataset(shuffle_data=False)

# 2. Split the data: 30% train, 20% validation, 50% test.
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data, 0.3, 0.2, 0.5)

# 3. Scale the data (both x and y) for each set.
x_train_scaled, y_train_scaled = scale_data(x_train, y_train)
x_val_scaled,   y_val_scaled   = scale_data(x_val, y_val)
x_test_scaled,  y_test_scaled  = scale_data(x_test, y_test)

# 4. Reshape the x arrays to 2D (Keras expects shape (n_samples, n_features))
x_train_scaled = x_train_scaled.reshape(-1, 1)
x_val_scaled   = x_val_scaled.reshape(-1, 1)
x_test_scaled  = x_test_scaled.reshape(-1, 1)

# 5. Build the model for Structure 1 using tanh activation:
#    - First hidden layer: 12 units with tanh activation
#    - Second hidden layer: 8 units with tanh activation
#    - Third hidden layer: 4 units with tanh activation
#    - Output layer: 1 unit for regression (linear activation by default)
model = Sequential([
    Dense(12, activation='tanh', input_shape=(1,)),
    Dense(8, activation='tanh'),
    Dense(4, activation='tanh'),
    Dense(1)
])

# 6. Compile the model using Adam optimizer and MSE loss.
model.compile(optimizer='adam', loss='mse')

# 7. Train the model for 20 epochs with a batch size of 12.
history = model.fit(x_train_scaled, y_train_scaled, epochs=20, batch_size=12,
                    validation_data=(x_val_scaled, y_val_scaled), verbose=1)

# 8. Predict on the test set.
y_pred_scaled = model.predict(x_test_scaled).flatten()

# 9. Calculate evaluation metrics on the scaled test data.
mae, mse_val, rmse, r2 = calculate_metrics(y_test_scaled, y_pred_scaled)
print("Metrics on Scaled Test Set (Case 5, Unshuffled Data):")
print("MAE =", mae)
print("MSE =", mse_val)
print("RMSE =", rmse)
print("R^2 =", r2)

# 10. Plot Actual vs. Predicted on the scaled test data.
plt.figure(figsize=(12, 5))

# Graph 1: Actual Scaled Test Data.
plt.subplot(1, 2, 1)
plt.scatter(x_test_scaled, y_test_scaled, color='blue', s=10, label='Actual')
plt.xlabel('x_test (scaled)')
plt.ylabel('y_test (scaled)')
plt.title('Actual Scaled Test Data (Unshuffled)')
plt.legend()

# Graph 2: Predicted Scaled Test Data.
plt.subplot(1, 2, 2)
plt.scatter(x_test_scaled, y_pred_scaled, color='red', s=10, label='Predicted')
plt.xlabel('x_test (scaled)')
plt.ylabel('y_pred (scaled)')
plt.title('Predicted Scaled Test Data (Unshuffled)')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np

# ---------------------------
# 1) Activation Functions
# ---------------------------
def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """
    Derivative of the sigmoid function.
    Given 'a' = sigmoid(z), the derivative w.r.t. z is:
    a * (1 - a).
    """
    return a * (1 - a)

# ---------------------------
# 2) XOR Data
# ---------------------------
# Inputs (A, B)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

# Labels (XOR output)
y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

# ---------------------------
# 3) Initialize Weights
# ---------------------------
# We'll have:
#   W1: (2 x 2) for input -> hidden
#   b1: (1 x 2)
#   W2: (2 x 1) for hidden -> output
#   b2: (1 x 1)

np.random.seed(0)  # for reproducible results (optional)

W1 = np.random.randn(2, 2) * 0.1
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.1
b2 = np.zeros((1, 1))

# ---------------------------
# 4) Hyperparameters
# ---------------------------
learning_rate = 0.1
num_epochs = 10000

# ---------------------------
# 5) Training Loop
# ---------------------------
for epoch in range(num_epochs):
    # ----- Forward Pass -----
    # Hidden layer
    z1 = np.dot(X, W1) + b1     # shape: (4, 2)
    a1 = sigmoid(z1)            # shape: (4, 2)

    # Output layer
    z2 = np.dot(a1, W2) + b2    # shape: (4, 1)
    a2 = sigmoid(z2)            # shape: (4, 1)  <-- predictions

    # ----- Compute Loss (MSE) -----
    # MSE = (1/4) * sum( (y - a2)^2 )
    # We'll keep the factor 1/4 implicit for gradient clarity
    loss = np.mean((y - a2)**2)

    # ----- Backpropagation -----
    # dL/da2 = -(y - a2)
    d_a2 = -(y - a2)  # shape: (4,1)

    # derivative of sigmoid for output
    d_z2 = d_a2 * sigmoid_derivative(a2)  # shape: (4,1)

    # Gradients for W2, b2
    # W2 gradient = a1^T * d_z2
    # b2 gradient = sum of d_z2 over all samples
    dW2 = np.dot(a1.T, d_z2)              # shape: (2,1)
    db2 = np.sum(d_z2, axis=0, keepdims=True)  # shape: (1,1)

    # Backprop to hidden layer
    # d_z1 = d_z2 * W2^T * sigmoid_derivative(a1)
    # But we need to do a matrix multiply:
    d_a1 = np.dot(d_z2, W2.T)            # shape: (4,2)
    d_z1 = d_a1 * sigmoid_derivative(a1) # shape: (4,2)

    # Gradients for W1, b1
    dW1 = np.dot(X.T, d_z1)              # shape: (2,2)
    db1 = np.sum(d_z1, axis=0, keepdims=True)  # shape: (1,2)

    # ----- Gradient Descent Update -----
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # (Optional) Print loss every so often
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.6f}")

# ---------------------------
# 6) Final Results
# ---------------------------
print("\nFinal training results:")
print(f"Weights W1:\n{W1}")
print(f"Bias b1:\n{b1}")
print(f"Weights W2:\n{W2}")
print(f"Bias b2:\n{b2}")

# ---------------------------
# 7) Test Predictions
# ---------------------------
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
predictions = sigmoid(z2)

print("\nXOR Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted (raw) = {predictions[i][0]:.4f}, "
          f"Rounded = {np.round(predictions[i][0])}, "
          f"Target = {y[i][0]}")