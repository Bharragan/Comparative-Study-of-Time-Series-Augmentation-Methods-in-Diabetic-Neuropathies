"""
Feature Extraction Library for Postural Time-Series Data

This module provides functions for extracting various features from 
center of pressure (COP) time-series data, such as acceleration, RMS, 
sample entropy, and frequency-based features.

Each function operates on a pandas DataFrame containing 'cop_x' and 'cop_y' columns, 
modifying the DataFrame in place by adding new feature columns.

Author: Nicolas Henriquez
"""


import numpy as np
import pandas as pd
from tqdm import tqdm

def add_acceleration_columns(df):
    """
    Adds acceleration columns for 'cop_x' and 'cop_y' series in the DataFrame.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'cop_x' and 'cop_y' columns.

    Returns:
    - None: Modifies the DataFrame in place, adding 'acc_x' and 'acc_y' columns.
    """
    acc_x_list = []
    acc_y_list = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc='Calculating accelerations'):
        x = np.array(row['cop_x'])
        y = np.array(row['cop_y'])
        
        acc_x = np.diff(x) / (0.04)
        acc_y = np.diff(y) / (0.04)
        
        acc_x = np.insert(acc_x, 0, 0)
        acc_y = np.insert(acc_y, 0, 0)
        
        acc_x_list.append(acc_x)
        acc_y_list.append(acc_y)

    df['acc_x'] = acc_x_list
    df['acc_y'] = acc_y_list

    

def add_path_column(df):
    """
    Adds the total path length for 'cop_x' and 'cop_y' series in the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing 'cop_x' and 'cop_y' columns.

    Returns:
    - None: Modifies the DataFrame in place, adding the 'path' column.
    """
    path_list = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc='Calculating path length'):
        x = np.array(row['cop_x'])
        y = np.array(row['cop_y'])
        path_length = sum(np.sqrt(x ** 2 + y ** 2))
        path_list.append(path_length)

    df['path'] = path_list



def add_rms_columns(df):
    """
    Adds RMS (Root Mean Square) values for 'acc_x' and 'acc_y' series in the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing 'acc_x' and 'acc_y' columns.

    Returns:
    - None: Modifies the DataFrame in place, adding 'rms_acc_x' and 'rms_acc_y' columns.
    """
    rms_x_list = []
    rms_y_list = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc='Calculating RMS'):
        x = np.array(row['acc_x'])
        y = np.array(row['acc_y'])
        rms_x = np.sqrt(np.mean(x ** 2))
        rms_y = np.sqrt(np.mean(y ** 2))
        rms_x_list.append(rms_x)
        rms_y_list.append(rms_y)

    df['rms_acc_x'] = rms_x_list
    df['rms_acc_y'] = rms_y_list


def add_sampen_columns(df, m=2, r=0.2):
    """
    Adds Sample Entropy for 'cop_x' and 'cop_y' series in the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing 'cop_x' and 'cop_y' columns.
    - m (int): Length of the subseries.
    - r (float): Similarity threshold.

    Returns:
    - None: Modifies the DataFrame in place, adding 'samp_en_x' and 'samp_en_y' columns.
    """
    
    def get_sampen_series(series, m, r):
        N = len(series)
        B = 0.0
        A = 0.0
        xmi = np.array([series[i: i + m] for i in range(N - m)])
        xmj = np.array([series[i: i + m] for i in range(N - m + 1)])
        B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
        m += 1
        xm = np.array([series[i: i + m] for i in range(N - m + 1)])
        A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
        return -np.log(A / B)

    for col in ['cop_x', 'cop_y']:
        tqdm.pandas(desc=f'Calculating Sample Entropy for {col}')
        df[f'samp_en_{col[-1]}'] = df[col].progress_apply(lambda series: get_sampen_series(series, m, r))

def add_f80_columns(df):
    """
    Adds F80 values (a frequency domain feature) for 'cop_x' and 'cop_y' series in the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing 'cop_x' and 'cop_y' columns.

    Returns:
    - None: Modifies the DataFrame in place, adding 'f80_x' and 'f80_y' columns.
    """
    
    def get_fft(serie):
        f = np.linspace(0.0, 1.0 / (2.0 * (1 / 25)), 500 // 2)
        power = np.abs(np.fft.fft(serie))[0:500 // 2]
        return f[f <= 4], power[f <= 4]

    def get_f80(f, power, min_val, max_val):
        return np.sum(power[(f >= min_val) & (f < max_val)]) * 4 / 5    

    def get_f80_on_column(serie):
        f, power = get_fft(serie)
        f80 = get_f80(f, power, 0, 4)
        return f80

    for col in ['cop_x', 'cop_y']:
        tqdm.pandas(desc=f'Calculating F80 for {col}')
        df[f'f80_{col[-1]}'] = df[col].progress_apply(get_f80_on_column)

def add_frequency_features(df):
    """
    Adds mean frequency features for different frequency bands for 'cop_x' and 'cop_y' series in the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing 'cop_x' and 'cop_y' columns.

    Returns:
    - None: Modifies the DataFrame in place, adding columns for mean frequencies for low, mid, and high frequency bands.
    """
    def get_fft(serie):
        f = np.linspace(0.0, 1.0 / (2.0 * (1 / 25)), 500 // 2)
        power = np.abs(np.fft.fft(serie))[0:500 // 2]
        return f[f <= 4], power[f <= 4]

    def get_mean_freq(f, power, min_freq, max_freq):
        mean = np.mean(power[(f >= min_freq) & (f < max_freq)])
        return mean

    def add_frequency_features_to_df(df, prefix, min_freq, max_freq):
        df[f'{prefix}_x'] = df['cop_x'].apply(lambda x: get_mean_freq(*get_fft(x), min_freq, max_freq))
        df[f'{prefix}_y'] = df['cop_y'].apply(lambda y: get_mean_freq(*get_fft(y), min_freq, max_freq))

    frequency_ranges = {
        'mf_lf': (0, 0.5),
        'mf_mf': (0.5, 2),
        'mf_hf': (2, 4)
    }

    for feature, (min_freq, max_freq) in tqdm(frequency_ranges.items(), desc='Calculating mean frequencies'):
        add_frequency_features_to_df(df, feature, min_freq, max_freq)




def add_rms_columns_cop(df):
    """
    Adds Root Mean Square (RMS) values for 'cop_x' and 'cop_y' series in the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing 'cop_x' and 'cop_y' columns.

    Returns:
    - None: Modifies the DataFrame in place, adding 'rms_x' and 'rms_y' columns.
    """
    
    rms_x_list = [] 
    rms_y_list = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc='Calculating RMS for Cop'):
        x = np.array(row['cop_x'])  
        y = np.array(row['cop_y']) 
        
        rms_x = np.sqrt(np.mean(x ** 2))
        rms_y = np.sqrt(np.mean(y ** 2))
        
        rms_x_list.append(rms_x) 
        rms_y_list.append(rms_y) 

    df['rms_x'] = rms_x_list 
    df['rms_y'] = rms_y_list 


def get_features(data):
    """
    Applies multiple feature extraction methods to the given DataFrame.

    Parameters:
    - data (DataFrame): The DataFrame to which the feature extraction methods will be applied.

    Returns:
    - DataFrame: The modified DataFrame with the new features added.
    """
    add_acceleration_columns(data)
    add_rms_columns(data)
    add_path_column(data)
    add_sampen_columns(data)
    add_f80_columns(data)
    add_frequency_features(data)
    add_rms_columns_cop(data)


    return data