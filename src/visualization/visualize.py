import matplotlib.pyplot as plt
import numpy as np
import os


def plot_original_and_augmented(original_series, augmented_series):
    """
    Plot the original and augmented time series.

    Parameters:
    original_series (array-like): Original time series data.
    augmented_series (array-like): Augmented time series data.

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(original_series, label='Original', color='blue')
    plt.plot(augmented_series, label='Aumentada', color='red', linestyle='--')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.title('Series de Tiempo Original y Aumentada')
    plt.legend()
    plt.show()
    
def plot_series(data, augmented_data, plot_original_first=True):
    """
    Plot multiple time series.

    Parameters:
    data (array-like): Original time series data.
    augmented_data (array-like): Augmented time series data.
    plot_original_first (bool): If True, plot original series first, otherwise plot augmented series first.

    Returns:
    None
    """
    fig, axs = plt.subplots(1, figsize=(18, 5))

    num_series = len(data)
    
    if plot_original_first:
        # Plot original time series data in blue
        for i in range(num_series):
            axs.plot(data[i], color='blue', label='Original' if i == 0 else "")
        
        # Plot augmented time series data in orange
        for i in range(num_series):
            axs.plot(augmented_data[i], color='orange', label='Augmented' if i == 0 else "")
    else:
        # Plot augmented time series data in orange
        for i in range(num_series):
            axs.plot(augmented_data[i], color='orange', label='Augmented' if i == 0 else "")
        
        # Plot original time series data in blue
        for i in range(num_series):
            axs.plot(data[i], color='blue', label='Original' if i == 0 else "")
    
    # Add legend
    axs.legend(loc='upper right')
    
    axs.set_xlabel('Tiempo')
    axs.set_ylabel('Valor')
    axs.set_title('Series de Tiempo Original y Aumentada')
    
    plt.tight_layout()
    plt.show()

def plot_cop_time_series(ts1, ts2, ts3, ts4, ts5, ts6, export_filename=None, output_dir='../reports/figures'):
    """
    Plots 6 time series in a 2x3 grid and optionally exports the plot as an image.

    Parameters:
    ts1, ts2, ts3, ts4, ts5, ts6 (np.ndarray): The six time series to be plotted.
    export_filename (str, optional): Filename for exporting the plot as an image (e.g., 'plot.png'). Default is None (no export).
    output_dir (str, optional): Directory where the plot image should be saved. Default is '../reports/figures'.

    Returns:
    None
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 7))
    fig.suptitle('Randomly Selected COP Time Series', fontsize=16)
    
    time = np.arange(len(ts1))
    
    titles = ['Artificial COP X Healthy Group', 'Artificial COP Y Healthy Group',
              'Artificial COP X Diabetic Group', 'Artificial COP Y Diabetic Group',
              'Artificial COP X Neuropathic Group', 'Artificial COP Y Neuropathic Group']
    
    series = [ts1, ts2, ts3, ts4, ts5, ts6]
    colors = ['g', 'g', 'b', 'b', 'orange', 'orange']
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(time, series[i], color=colors[i], linestyle='-', marker='o', markersize=2)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_title(titles[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar la imagen si se especifica un nombre de archivo para exportación
    if export_filename:
        plt.savefig(os.path.join(output_dir, export_filename))
    
    plt.show()


def plot_cop_and_acc_time_series(df, export_filename=None, alt_title='COP and ACC Time Series', output_dir='../reports/figures'):
    """
    Plots COP and ACC time series for the given DataFrame and optionally exports the plot as an image.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'cop_x', 'cop_y', 'acc_x', 'acc_y', and 'class' columns.
    export_filename (str, optional): Filename for exporting the plot as an image (e.g., 'plot.png'). Default is None (no export).
    alt_title (str, optional): Alternative title for the plot. Default is 'COP and ACC Time Series'.
    output_dir (str, optional): Directory where the plot image should be saved. Default is 'plots'.

    Returns:
    None
    """
    # Filtrar las clases
    healthy = df[df['class'] == 'Healthy']
    diabetic = df[df['class'] == 'Diabetic']
    neuropathic = df[df['class'] == 'Neuropathic']
    
    # Crear subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle(alt_title, fontsize=16)
    
    time = np.arange(len(healthy.iloc[0]['cop_x']))
    
    # Títulos y datos de las series
    titles = ['COP X', 'COP Y', 'ACC X', 'ACC Y']
    data = [(healthy['cop_x'], diabetic['cop_x'], neuropathic['cop_x']),
            (healthy['cop_y'], diabetic['cop_y'], neuropathic['cop_y']),
            (healthy['acc_x'], diabetic['acc_x'], neuropathic['acc_x']),
            (healthy['acc_y'], diabetic['acc_y'], neuropathic['acc_y'])]
    
    colors = ['g', 'b', 'orange']
    labels = ['Healthy', 'Diabetic', 'Neuropathic']
    
    # Graficar cada subplot
    for i, ax in enumerate(axes.flatten()):
        for j, (series, color) in enumerate(zip(data[i], colors)):
            for s in series:
                ax.plot(time, s, color=color, alpha=0.7 if color != 'g' else 1.0, zorder=3-j)
        ax.set_title(titles[i])
        ax.set_xlabel('Sampled time')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
    
    handles = [plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10) for color in colors]
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar la imagen si se especifica un nombre de archivo para exportación
    if export_filename:
        plt.savefig(os.path.join(output_dir, export_filename))
    
    plt.show()

    
def plot_real_artificial_data(ts_real1, ts_real2, ts_real3, ts_artificial1, ts_artificial2, ts_artificial3, export_filename=None, output_dir='../reports/figures', alt_title=None):
    """
    Plots 6 time series (3 real and 3 artificial) in a 2x3 grid and optionally exports the plot as an image.

    Parameters:
    ts_real1, ts_real2, ts_real3 (np.ndarray): The three real time series to be plotted.
    ts_artificial1, ts_artificial2, ts_artificial3 (np.ndarray): The three artificial time series to be plotted.
    export_filename (str, optional): Filename for exporting the plot as an image (e.g., 'plot.png'). Default is None (no export).
    output_dir (str, optional): Directory where the plot image should be saved. Default is '../reports/figures'.
    alt_title (str, optional): Alternative title for the plot. Default is None.

    Returns:
    None
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 7))
    if alt_title:
        fig.suptitle(alt_title, fontsize=16)
    else:
        fig.suptitle('Real and Artificial Data Time Series', fontsize=16)
    
    time_real1 = np.arange(len(ts_real1))
    time_real2 = np.arange(len(ts_real2))
    time_real3 = np.arange(len(ts_real3))
    time_artificial1 = np.arange(len(ts_artificial1))
    time_artificial2 = np.arange(len(ts_artificial2))
    time_artificial3 = np.arange(len(ts_artificial3))
    
    titles = ['Healthy Real Data', 'Healthy Artificial Data',
              'Diabetic Real Data', 'Diabetic Artificial Data',
              'Neuropathic Real Data', 'Neuropathic Artificial Data']
    
    series = [ts_real1, ts_artificial1, ts_real2, ts_artificial2, ts_real3, ts_artificial3]
    colors = ['g', 'g', 'b', 'b', 'orange', 'orange']
    
    times = [time_real1, time_artificial1, time_real2, time_artificial2, time_real3, time_artificial3]

    for i, ax in enumerate(axes.flatten()):
        ax.plot(times[i], series[i], color=colors[i], linestyle='-', marker='o', markersize=2)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_title(titles[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar la imagen si se especifica un nombre de archivo para exportación
    if export_filename:
        plt.savefig(os.path.join(output_dir, export_filename))
    
    plt.show()