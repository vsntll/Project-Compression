import os
import cv2
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


# Import from your new self-contained fractal codec
from src.fractal_codec_structured import fractal_compress, fractal_decompress, fractal_save

def estimate_bitrate(filesize_bytes, img_shape):
    """Calculates bitrate in bits per pixel."""
    num_pixels = img_shape[0] * img_shape[1]
    return (filesize_bytes * 8) / num_pixels

def process_image_task(args):
    """
    A self-contained function to process a single image.
    This function is designed to be run in a separate process.
    """
    image_path, dataset_name, dataset_output_dir, worker_id = args
    image_name = os.path.basename(image_path)

    # 1. Read the image
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 2. Compress the image, passing the worker_id as the position for the progress bar
    compressed_data = fractal_compress(img_color, show_progress=True, position=worker_id)

    # 3. Save compressed data to measure file size
    fractal_file_path = os.path.join(dataset_output_dir, f"{os.path.splitext(image_name)[0]}_fractal.pkl")
    fractal_save(compressed_data, fractal_file_path)

    # 4. Decompress the image
    reconstructed_img = fractal_decompress(compressed_data)

    # 5. Save the reconstructed image
    output_path = os.path.join(dataset_output_dir, f"fractal_recon_{image_name}")
    cv2.imwrite(output_path, reconstructed_img)

    # 6. Calculate and return metrics
    metrics = {
        'dataset': dataset_name,
        'image': image_name,
        'psnr': psnr(img_gray, reconstructed_img, data_range=255),
        'ssim': ssim(img_gray, reconstructed_img, data_range=255),
        'bitrate': estimate_bitrate(os.path.getsize(fractal_file_path), img_gray.shape)
    }
    return metrics

def run_fractal_structured_test(data_dirs, output_csv_name):
    """
    Tests the structured fractal codec on all images in the provided directories
    and saves the reconstructed images and metrics.
    """
    # --- Configuration ---
    base_results_dir = "results"
    output_image_dir = os.path.join(base_results_dir, "fractal_structured_dfs_parallel")
    metrics_dir = "results/metrics/"

    # --- Setup ---
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_image_dir}")
    print(f"Metrics will be saved to: {metrics_dir}")

    collected_metrics = []
    tasks = []

    # --- Prepare all tasks for all datasets ---
    for dataset_name, dataset_path in data_dirs.items():
        try:
            image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            if not image_files:
                print(f"Warning: No images found in '{dataset_path}'. Skipping.")
                continue

            dataset_output_dir = os.path.join(output_image_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            for image_name in image_files:
                image_path = os.path.join(dataset_path, image_name)
                tasks.append((image_path, dataset_name, dataset_output_dir)) # worker_id will be added later

        except FileNotFoundError:
            print(f"Error: Input directory not found at '{dataset_path}'. Skipping.")
            continue

    # --- Execute tasks in parallel ---
    print(f"\nStarting parallel processing of {len(tasks)} images...")
    # Let the executor decide the number of workers (usually os.cpu_count())
    max_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Assign a worker_id to each task argument tuple
        # The position will be from 1 to max_workers, leaving position 0 for the main bar
        tasks_with_ids = [(*task, (i % max_workers) + 1) for i, task in enumerate(tasks)]

        # Main progress bar showing overall image completion
        pbar = tqdm(total=len(tasks), desc="Overall Progress", unit="image", position=0)

        # Submit tasks and collect results as they complete
        futures = [executor.submit(process_image_task, task) for task in tasks_with_ids]
        for future in as_completed(futures):
            pbar.update(1)
            try:
                metrics = future.result()
                collected_metrics.append(metrics)
            except Exception as exc:
                print(f'Task generated an exception: {exc}')

    # --- Save Metrics to CSV ---
    # Sort by dataset and image name for consistent output
    collected_metrics.sort(key=lambda m: (m['dataset'], m['image']))
    df = pd.DataFrame(collected_metrics)
    csv_output_path = os.path.join(metrics_dir, "fractal_structured_dfs_parallel.csv")
    df.to_csv(csv_output_path, index=False)

    print(f"\nTest complete. Reconstructed images are saved in '{output_image_dir}'.")
    print(f"Metrics saved to '{csv_output_path}'.")

if __name__ == "__main__":
    data_dirs = {
        "kodak": "data/kodak",
        "standard_test": "data/standard_test",
        "clic_subset": "data/clic_subset"
    }
    run_fractal_structured_test(data_dirs, "fractal_structured_dfs_parallel.csv")