import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_comparison_curves(df, output_dir, datasets, x_axis='bitrate'):
    """
    Generates and saves Rate-Distortion (PSNR/SSIM vs. Bitrate/FileSize) plots.
    """
    y_metrics = ['psnr', 'ssim']

    for metric in y_metrics:
        for dataset in datasets:
            plt.figure(figsize=(12, 8))
            dataset_df = df[df['dataset'] == dataset]

            if dataset_df.empty:
                continue

            if x_axis == 'filesize':
                # Handle filesize plot with its specific column names
                plt.scatter(dataset_df['filesize_wavelet_kb'], dataset_df[f'{metric}_wavelet'], label='Wavelet', marker='x', color='blue')
                plt.scatter(dataset_df['filesize_fractal_kb'], dataset_df[f'{metric}_fractal'], label='Fractal', marker='o', color='red', alpha=0.7)
            else: # Default behavior for bitrate
                # Plot Wavelet data
                plt.scatter(dataset_df[f'{x_axis}_wavelet'], dataset_df[f'{metric}_wavelet'], label='Wavelet', marker='x', color='blue')
                # Plot Fractal data
                plt.scatter(dataset_df[f'{x_axis}_fractal'], dataset_df[f'{metric}_fractal'], label='Fractal', marker='o', color='red', alpha=0.7)

            # Formatting the plot
            metric_name = metric.upper()
            x_label = 'Bitrate (bits per pixel)' if x_axis == 'bitrate' else 'File Size (KB)'
            y_label = f'{metric_name} (dB)' if metric_name == 'PSNR' else metric_name

            plt.title(f'{metric_name} vs. {x_label.split(" (")[0]} for "{dataset}" Dataset')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, which='both', linestyle='--')
            plt.legend()

            # Save the plot
            plot_filename = os.path.join(output_dir, f'{dataset}_{metric}_vs_{x_axis}.png')
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
            plt.close()

def plot_single_image_comparison(df, image_name, output_dir):
    """
    Generates a bar chart comparing all metrics for a single image.
    """
    image_data = df[df['image'] == image_name]
    if image_data.empty:
        print(f"\nWarning: Image '{image_name}' not found in the metrics file. Skipping single image plot.")
        return

    metrics = image_data.iloc[0]
    labels = ['PSNR (dB)', 'SSIM', 'Bitrate (bpp)', 'File Size (KB)']
    wavelet_values = [metrics['psnr_wavelet'], metrics['ssim_wavelet'], metrics['bitrate_wavelet'], metrics['filesize_wavelet_kb']]
    fractal_values = [metrics['psnr_fractal'], metrics['ssim_fractal'], metrics['bitrate_fractal'], metrics['filesize_fractal_kb']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, wavelet_values, width, label='Wavelet', color='blue')
    rects2 = ax.bar(x + width/2, fractal_values, width, label='Fractal', color='red')

    ax.set_ylabel('Scores / Values')
    ax.set_title(f'Metric Comparison for Image: {image_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plot_filename = os.path.join(output_dir, f'{image_name}_metrics_comparison.png')
    plt.savefig(plot_filename)
    print(f"\nSaved single image comparison plot: {plot_filename}")
    plt.close()

def generate_summary_table(df, output_csv_path):
    """
    Prints a summary table and saves it to a CSV file.
    """
    summary_data = []
    for index, row in df.iterrows():
        summary = {'dataset': row['dataset'], 'image': row['image']}
        # Higher is better for PSNR and SSIM
        summary['psnr_winner'] = 'Wavelet' if row['psnr_wavelet'] > row['psnr_fractal'] else 'Fractal'
        summary['ssim_winner'] = 'Wavelet' if row['ssim_wavelet'] > row['ssim_fractal'] else 'Fractal'
        # Lower is better for bitrate and file size
        summary['bitrate_winner'] = 'Wavelet' if row['bitrate_wavelet'] < row['bitrate_fractal'] else 'Fractal'
        summary['filesize_winner'] = 'Wavelet' if row['filesize_wavelet_kb'] < row['filesize_fractal_kb'] else 'Fractal'
        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)
    print("\n--- Compression Method Performance Summary ---")
    print("This table shows which method performed better for each metric on a per-image basis.")
    print("(Higher is better for PSNR/SSIM; Lower is better for Bitrate/FileSize)\n")
    print(summary_df.to_string())

    # Save the summary dataframe to a CSV file
    summary_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved performance summary to: {output_csv_path}")

def main(wavelet_csv, fractal_csv, output_dir, summary_csv_path, image_for_single_plot):
    """
    Main function to run all plotting and analysis.
    """
    try:
        df_wavelet = pd.read_csv(wavelet_csv)
        df_fractal = pd.read_csv(fractal_csv)
    except FileNotFoundError:
        print(f"Error: One or both metrics files not found.")
        print(f"Please ensure '{wavelet_csv}' and '{fractal_csv}' exist.")
        return

    # Rename columns to distinguish between the methods
    df_wavelet = df_wavelet.rename(columns={
        'psnr': 'psnr_wavelet', 'ssim': 'ssim_wavelet', 'bitrate': 'bitrate_wavelet'
    })
    df_fractal = df_fractal.rename(columns={
        'psnr': 'psnr_fractal', 'ssim': 'ssim_fractal', 'bitrate': 'bitrate_fractal'
    })

    # Merge the two dataframes on image and dataset
    df = pd.merge(df_wavelet, df_fractal, on=['dataset', 'image'])
    # --- Calculate File Size from Bitrate ---
    # Bitrate = (filesize_bytes * 8) / (width * height)
    # Filesize_bytes = (Bitrate * width * height) / 8
    # We need image dimensions. Let's assume a standard size if not available,
    # or parse from the image name if possible. For kodak, it's 768x512.
    # A more robust solution would be to save dimensions in the CSV.
    # For this example, we'll assume Kodak dimensions.
    # This is a simplification; main.py should be updated to save dimensions.
    print("\nWarning: Assuming image dimensions (e.g., 768x512 for Kodak) to calculate file size.")
    print("For accurate results, modify 'main.py' to save image dimensions in the CSV.\n")
    
    # Example dimensions for Kodak dataset
    width, height = 768, 512
    num_pixels = width * height

    df['filesize_wavelet_kb'] = (df['bitrate_wavelet'] * num_pixels) / (8 * 1024)
    df['filesize_fractal_kb'] = (df['bitrate_fractal'] * num_pixels) / (8 * 1024)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir}")

    datasets = df['dataset'].unique()

    # --- Generate Plots ---
    print("\n--- Generating PSNR/SSIM vs. Bitrate plots ---")
    plot_comparison_curves(df, output_dir, datasets, x_axis='bitrate')

    print("\n--- Generating PSNR/SSIM vs. File Size plots ---")
    plot_comparison_curves(df, output_dir, datasets, x_axis='filesize')

    # --- Generate Single Image Plot ---
    if image_for_single_plot:
        plot_single_image_comparison(df, image_for_single_plot, output_dir)

    # --- Print Summary Table ---
    generate_summary_table(df, summary_csv_path)

if __name__ == "__main__":
    # CSV files generated by the compression scripts
    wavelet_metrics_file = 'results/metrics/wavelet.csv'
    fractal_metrics_file = 'results/metrics/fractal_cuda_parallel.csv'

    results_dir = 'results'
    plot_dir = 'results/detailed_plots/'
    summary_output_file = os.path.join(results_dir, 'performance_summary.csv')

    # Specify an image name from your dataset for the detailed bar chart comparison.
    # Example: 'kodim05.png'. Set to None to skip this plot.
    single_image_to_plot = 'kodim05.png'

    main(wavelet_metrics_file, fractal_metrics_file, plot_dir, summary_output_file, single_image_to_plot)
