import numpy as np

def create_training_data(ts_data: np.ndarray, num_features: int) -> tuple[np.ndarray]:
    """
    Create a training dataset from a 1D time series array.

    Args:
        ts_data (np.ndarray): 1D array representing the time series data.
        num_features (int): Number of features (time steps) to use for each sample.

    Returns:
        tuple[np.ndarray]: A tuple containing the training matrix X and the target vector y.

    Raises:
        None.

    Example:
        ts_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        num_features = 3
        X, y = create_training_data(ts_data, num_features)
    """
    num_samples = len(ts_data) - num_features
    
    # Create the indices for slicing the data
    indices = np.arange(num_samples).reshape(-1, 1) + np.arange(num_features)
    
    # Create the X matrix by slicing the data
    X = ts_data[indices.flatten()].reshape(num_samples, num_features)
    
    # Create the y vector by slicing the data
    y = ts_data[num_features:]
    
    return X, y
