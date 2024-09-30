"""
Data Augmentation Library for Postural Time-Series Data

This module provides functions for augmenting time series data.

Author: Nicolas Henriquez
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import check_random_state
import src.utils.dtw as dtw


def scaling(X, num_samples=1, scale_factors=[0.7, 0.8, 0.9, 1.1, 1.2, 1.3]):
    """
    Augment a set of time series by applying random scaling.

    Parameters:
    X (list): The original set of time series.
    num_samples (int): The number of scaled time series samples to generate.
                       Default is 1.
    scale_factors (list): List of scaling factors to choose from randomly.
                          Default is [0.7, 0.8, 0.9, 1.1, 1.2, 1.3].

    Returns:
    augmented_data (list): The set of augmented time series.
    """
    X = np.array([np.array(series) for series in X])

    augmented_data = []
    current_samples = len(X)
    total_samples_needed = num_samples - current_samples
    
    if total_samples_needed > 0:
        while total_samples_needed > 0:
            series_idx = np.random.randint(0, current_samples) 
            scale_factor = np.random.choice(scale_factors)     
            augmented_series = X[series_idx] * scale_factor 
            augmented_data.append(augmented_series)
            total_samples_needed -= 1

    return augmented_data


def shuffle_time_slices(time_series, slice_size):
    """
    Shuffle different time slices of the provided array.

    Parameters:
    time_series (array-like): An array containing time-series data.
    slice_size (int): The size of each time slice that will be shuffled.

    Returns:
    shuffled_data (array-like): The array with shuffled time slices.
    """
    time_series = np.array(time_series)

    if slice_size <= 0 or slice_size > len(time_series):
        raise ValueError("Slice size must be within the range 1 to len(data)")

    num_slices = len(time_series) // slice_size

    slices = [time_series[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)]

    np.random.shuffle(slices)

    shuffled_data = np.concatenate(slices)

    remainder = len(time_series) % slice_size
    if remainder > 0:
        remainder_data = time_series[-remainder:]
        shuffled_data = np.concatenate([shuffled_data, remainder_data])

    return shuffled_data

def random_shuffling(time_series_set, num_samples, slice_size=100):
    """
    Augment time series data by applying random shuffling of time slices.

    Parameters:
    time_series_set (array-like): The set of time series data.
    num_samples (int): Number of artificial samples to generate.
    slice_size (int): Size of each time slice to shuffle.

    Returns:
    augmented_data (array-like): Augmented time series data.
    """
    time_series_set = np.array([np.array(series) for series in time_series_set])

    augmented_data = []

    min_samples_needed = max(num_samples, 1600)

    for _ in range(min_samples_needed):
        idx = np.random.randint(len(time_series_set))  # Randomly select a time series
        time_series = time_series_set[idx]
        shuffled_slices = shuffle_time_slices(time_series, slice_size=slice_size)
        augmented_data.append(shuffled_slices)

    return augmented_data



def detect_outliers(series, threshold=3):
    """
    Detect outliers in a time series using standard deviation.

    Parameters:
    series (np.array): The time series.
    threshold (float): The threshold multiplier for the standard deviation.

    Returns:
    boolean: True if outliers are detected, False otherwise.
    """
    std = np.std(series)
    return any(abs(series - np.mean(series)) > threshold * std)

def window_warping(X, num_samples=1600, window_size_ratio=0.5, scale_factors=[0.9, 1.1], outlier_threshold=3):
    """
    Augment a set of time series using window warping with outlier filtering.

    Parameters:
    X (list): The original set of time series.
    num_samples (int): Approximate number of artificial samples to generate.
    window_size_ratio (float): Ratio of the window size to the total length of the time series.
    scale_factors (list): List of scaling factors for window warping.
    outlier_threshold (float): Threshold for detecting outliers.

    Returns:
    augmented_data (list): The set of augmented time series.
    """
    augmented_data = []
    
    X_filtered = [series for series in X if not detect_outliers(series, outlier_threshold)]
    
    num_repeats = int(np.ceil(num_samples / len(X_filtered)))
    
    for _ in range(num_repeats):
        for series in X_filtered:
            series_length = len(series)
            window_size = int(series_length * window_size_ratio)
            window_size = max(window_size, 1)
            
            start_idx = np.random.randint(0, series_length - window_size + 1)
            scale_factor = np.random.choice(scale_factors)
            
            window = series[start_idx:start_idx + window_size]
            warped_window = np.interp(
                np.linspace(0, window_size, int(window_size * scale_factor)),
                np.arange(window_size),
                window
            )
            
            augmented_series = np.concatenate([
                series[:start_idx],
                warped_window,
                series[start_idx + window_size:]
            ])
            
            if len(augmented_series) > series_length:
                augmented_series = augmented_series[:series_length]
            elif len(augmented_series) < series_length:
                augmented_series = np.pad(augmented_series, (0, series_length - len(augmented_series)), 'constant')
            
            augmented_data.append(augmented_series)
            
            if len(augmented_data) >= num_samples:
                break
        
        if len(augmented_data) >= num_samples:
            break
    
    return augmented_data

def add_gaussian_noise(time_series, sigma_range=(0.02, 0.04)):
    """
    Add ranged Gaussian noise to a time series.

    Parameters:
    time_series (list or np.ndarray): The original time series.
    sigma_range (tuple): Standard deviation range for the Gaussian noise.
                         Default is (0.02, 0.04).

    Returns:
    np.ndarray: The time series with added Gaussian noise.
    """
    time_series = np.array(time_series)
    
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    
    return time_series + np.random.normal(loc=0., scale=sigma, size=time_series.shape)


def augment_time_series_with_noise(X, num_samples=1):
    """
    Augment a set of time series by adding Gaussian noise multiple times.

    Parameters:
    X (list of lists or np.ndarray): The set of original time series.
    num_samples (int): Total number of noisy samples to generate.
                       Default is 1.

    Returns:
    augmented_data (list): Set of time series with added Gaussian noise.
    """
    samples_per_series = int(np.ceil(num_samples / len(X)))    
    augmented_data = []
    
    for series in X:
        for _ in range(samples_per_series):
            noisy_series = add_gaussian_noise(series)
            augmented_data.append(noisy_series)
    
    return augmented_data



def flip_time_series_set(X):
    """
    Flip a set of time series.

    Parameters:
    X (list): Set of original time series.

    Returns:
    flipped_data (list): Set of flipped time series.
    """
    flipped_data = []
    
    for series in X:
        flipped_series = np.flip(series)
        flipped_data.append(flipped_series)
    
    return flipped_data


def jitter(x, sigma=0.03):
    """Add Simple Gaussian noise."""
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def spawner(x, labels, num_samples, sigma=0.05, verbose=0, random_state=None):
    """
    Generate augmented data using Dynamic Time Warping (DTW) and interpolation.

    This function is a modified version based on the method described in the paper:
    "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks"
    (https://arxiv.org/abs/2007.15951).
    
    The original method on which this technique is based was proposed in:
    "Data Augmentation with Suboptimal Warping for Time-Series Classification"
    (https://ieeexplore.ieee.org/document/8215569).
    
    The function generates new samples by selecting random points in time series, applying 
    Dynamic Time Warping to align portions of two time series with similar labels, 
    interpolating between them, and adding random noise.

    Parameters:
    - x: np.ndarray
        Original dataset, a 3D array with dimensions (n_samples, n_timestamps, n_features).
    - labels: np.ndarray
        Corresponding labels for the dataset. Can be one-hot encoded or a single array.
    - num_samples: int
        The number of augmented samples to generate.
    - sigma: float, default=0.05
        Standard deviation for jitter noise added to the generated samples.
    - verbose: int, default=0
        Level of verbosity. 0 for no output, higher values for more detailed logs.
    - random_state: int, RandomState instance, or None, default=None
        Seed or RandomState instance for reproducibility.

    Returns:
    - ret: np.ndarray
        Augmented dataset of shape (num_samples, n_timestamps, n_features).
    """
    random_state = check_random_state(random_state)
    n_samples, n_timestamps, n_features = x.shape
    orig_steps = np.arange(n_timestamps)
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros((num_samples, n_timestamps, n_features))
    
    for i in tqdm(range(num_samples), desc="Generating Data"):
        idx = random_state.randint(n_samples)
        random_points = np.random.randint(low=1, high=n_timestamps - 1, size=1)[0]
        window = np.ceil(n_timestamps / 10.).astype(int)
        choices = np.delete(np.arange(n_samples), idx)
        choices = np.where(l[choices] == l[idx])[0]
        
        if choices.size > 0:
            random_sample = x[np.random.choice(choices)]
            path1 = dtw.dtw(x[idx][:random_points], random_sample[:random_points], return_flag=dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(x[idx][random_points:], random_sample[random_points:], return_flag=dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2 + random_points)), axis=1)

            mean = np.mean([x[idx][combined[0]], random_sample[combined[1]]], axis=0)
            
            for dim in range(n_features):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, n_timestamps - 1., num=mean.shape[0]), mean[:, dim]).T
        else:
            ret[i, :] = x[idx]
    
    return jitter(ret, sigma=sigma)
def random_guided_warp(x, labels, num_samples, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    """
    Generate augmented time series data using guided warping based on Dynamic Time Warping (DTW).

    This is a modified version of the code from the paper:
    "Data Augmentation with Suboptimal Warping for Time-Series Classification" (https://arxiv.org/abs/2007.15951).

    Parameters:
    - x : numpy.ndarray
        The input time series data of shape (n_samples, n_timestamps, n_features).
    - labels : numpy.ndarray
        The labels for the time series data, with shape (n_samples,) or (n_samples, n_classes).
    - num_samples : int
        The number of augmented samples to generate.
    - slope_constraint : str, optional
        The slope constraint for DTW, either "symmetric" or "asymmetric". Default is "symmetric".
    - use_window : bool, optional
        Whether to use a window for DTW. Default is True.
    - dtw_type : str, optional
        The type of DTW to use: "normal" for standard DTW, "shape" for shapeDTW. Default is "normal".
    - verbose : int, optional
        Level of verbosity; use -1 to suppress warnings. Default is 0.

    Returns:
    - numpy.ndarray
        The augmented time series data of shape (num_samples, n_timestamps, n_features).
    """
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros((num_samples, x.shape[1], x.shape[2]))
    for i in tqdm(range(num_samples)):
        # Randomly select an index from the dataset
        index = np.random.randint(0, x.shape[0])
        pat = x[index]
        # Guarantees that the same one isn't selected
        choices = np.delete(np.arange(x.shape[0]), index)
        # Remove ones of different classes
        choices = choices[l[choices] == l[index]]
        if choices.size > 0:        
            # Pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]
            
            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                            
            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping"%l[index])
            ret[i,:] = pat
    return ret
