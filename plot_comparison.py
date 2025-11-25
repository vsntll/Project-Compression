import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_rd_curves(fractal_csv, wavelet_csv, output_dir):
    """
    Generates and saves Rate-Distortion (PSNR vs. Bitrate) plots
    to compare fractal and wavelet compression performance.
    """
    try:
        # The main.py output contains fractal data
        df_fractal = pd.read_csv(fractal_csv)
        # The wavelet_test.py output contains wavelet data
        df_wavelet = pd.read_csv(wavelet_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find a metrics file. {e}")
        print("Please run 'python main.py' and 'python wavelet_test.py' first to generate the CSV files.")
        return

    # Rename columns for clarity and consistency before merging/plotting
    df_fractal = df_fractal.rename(columns={
        'bitrate_fractal': 'bitrate',
        'psnr_fractal': 'psnr'
    })

    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir}")

    # Get the unique datasets from both files
    datasets = pd.concat([df_fractal['dataset'], df_wavelet['dataset']]).unique()

    for dataset in datasets:
        plt.figure(figsize=(10, 7))

        # Filter data for the current dataset
        fractal_data = df_fractal[df_fractal['dataset'] == dataset]
        wavelet_data = df_wavelet[df_wavelet['dataset'] == dataset]

        # Plotting
        plt.scatter(fractal_data['bitrate'], fractal_data['psnr'], label='Fractal', marker='o')
        plt.scatter(wavelet_data['bitrate'], wavelet_data['psnr'], label='Wavelet', marker='x')

        # Formatting the plot
        plt.title(f'PSNR vs. Bitrate Comparison for "{dataset}" Dataset')
        plt.xlabel('Bitrate (bits per pixel)')
        plt.ylabel('PSNR (dB)')
        plt.grid(True, which='both', linestyle='--')
        plt.legend()

        # Save the plot
        plot_filename = os.path.join(output_dir, f'{dataset}_comparison.png')
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

if __name__ == "__main__":
    # Assumes main.py has been run to generate fractal metrics
    fractal_metrics_file = 'results/metrics/fractal.csv'
    wavelet_metrics_file = 'results/metrics/wavelet.csv'
    plot_output_dir = 'results/plots/'

    plot_rd_curves(fractal_metrics_file, wavelet_metrics_file, plot_output_dir)