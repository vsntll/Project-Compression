import os
import cv2
import pandas as pd
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


# Import from the CUDA-accelerated fractal codec
from src.fractal_codec_cuda import fractal_compress_cuda, fractal_decompress_cuda, fractal_save
# Import the CPU-based codec for hybrid processing
from src.fractal_codec_structured import fractal_compress as fractal_compress_cpu, fractal_decompress as fractal_decompress_cpu

def _find_and_set_cuda_path():
    """
    Tries to find the CUDA Toolkit path and set the CUDA_PATH environment variable.
    This helps CuPy locate the necessary CUDA libraries, especially in new processes.
    """
    # Search for the CUDA Toolkit in the default installation directory
    possible_paths = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
    if possible_paths:
        # Use the latest version found
        latest_cuda_path = sorted(possible_paths)[-1]

        # 1. Set CUDA_PATH environment variable
        if 'CUDA_PATH' not in os.environ:
            print(f"Automatically setting CUDA_PATH to: {latest_cuda_path}")
            os.environ['CUDA_PATH'] = latest_cuda_path

        # 2. Add CUDA /bin directory to the system PATH for DLL loading
        cuda_bin_path = os.path.join(latest_cuda_path, 'bin')
        if cuda_bin_path not in os.environ['PATH']:
            print(f"Adding CUDA bin to system PATH: {cuda_bin_path}")
            os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']

def estimate_bitrate(filesize_bytes, img_shape):
    """Calculates bitrate in bits per pixel."""
    num_pixels = img_shape[0] * img_shape[1]
    return (filesize_bytes * 8) / num_pixels

def process_image_gpu_task(args):
    """
    A self-contained function to process a single image on the GPU.
    """
    image_path, dataset_name, dataset_output_dir, worker_id = args
    worker_id += 2 # Offset for dataset and image progress bars
    image_name = os.path.basename(image_path)

    # 1. Read the image
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 2. Compress the image, passing the worker_id as the position for the progress bar
    compressed_data = fractal_compress_cuda(img_color, show_progress=True, position=worker_id, distortion_threshold=0.001)

    # 3. Save compressed data to measure file size
    fractal_file_path = os.path.join(dataset_output_dir, f"{os.path.splitext(image_name)[0]}_fractal.pkl")
    fractal_save(compressed_data, fractal_file_path)

    # 4. Decompress the image
    reconstructed_img = fractal_decompress_cuda(compressed_data)

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

def process_image_cpu_task(args):
    """
    A self-contained function to process a single image on the CPU.
    """
    image_path, dataset_name, dataset_output_dir, worker_id = args
    worker_id += 2 # Offset for dataset and image progress bars
    image_name = os.path.basename(image_path)

    # 1. Read the image
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 2. Compress the image using the CPU codec
    compressed_data = fractal_compress_cpu(img_color, show_progress=True, position=worker_id, distortion_threshold=0.005)

    # 3. Save compressed data to measure file size
    fractal_file_path = os.path.join(dataset_output_dir, f"{os.path.splitext(image_name)[0]}_fractal_cpu.pkl")
    fractal_save(compressed_data, fractal_file_path)

    # 4. Decompress the image
    reconstructed_img = fractal_decompress_cpu(compressed_data)

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
    output_image_dir = os.path.join(base_results_dir, "fractal_cuda_parallel")
    metrics_dir = "results/metrics/"

    # --- Setup ---
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_image_dir}")
    print(f"Metrics will be saved to: {metrics_dir}")

    collected_metrics = []
    
    # --- Execute tasks dataset by dataset to have per-dataset progress bars ---
    print(f"\nStarting parallel processing across {len(data_dirs)} datasets...")
    
    # --- Hybrid Worker Allocation ---
    # To cap GPU utilization, we limit the number of processes that can submit work to the GPU.
    # The remaining CPU cores will work on CPU-based compression in parallel.
    max_workers = os.cpu_count()
    # Set a hard cap on concurrent GPU processes to control utilization. '2' is a good starting point.
    NUM_GPU_WORKERS = min(2, max_workers - 1 if max_workers > 1 else 1) # Ensure at least 1 CPU worker if possible
    num_cpu_workers = max_workers - NUM_GPU_WORKERS

    print(f"Allocating {NUM_GPU_WORKERS} worker(s) for GPU tasks and {num_cpu_workers} for CPU tasks to keep GPU utilization below 100%.")
    # Outer loop for datasets, with its own progress bar
    dataset_pbar = tqdm(data_dirs.items(), desc="Total Datasets", unit="dataset", position=0)
    for dataset_name, dataset_path in dataset_pbar:
        dataset_pbar.set_description(f"Processing Dataset: {dataset_name}")

        tasks = []
        # --- Prepare tasks for the current dataset ---
        try: 
            image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            if not image_files: 
                print(f"Warning: No images found in '{dataset_path}'. Skipping.")
                continue

            dataset_output_dir = os.path.join(output_image_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            for image_name in image_files:
                image_path = os.path.join(dataset_path, image_name)
                tasks.append((image_path, dataset_name, dataset_output_dir))

        except FileNotFoundError:
            print(f"Error: Input directory not found at '{dataset_path}'. Skipping.")
            continue

        # --- Split tasks for CPU and GPU ---
        # Dynamically split work based on the number of workers.
        # This is a heuristic to balance the load, assuming GPU is much faster.
        total_tasks = len(tasks)
        split_index = min(total_tasks, int(total_tasks * (NUM_GPU_WORKERS / max_workers) * 5) if max_workers > 0 else 0)
        gpu_tasks = tasks[:split_index] 
        cpu_tasks = tasks[split_index:]

        # --- Execute tasks for the current dataset in parallel ---
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            # Submit GPU tasks
            if gpu_tasks:
                gpu_tasks_with_ids = [(*task, (i % NUM_GPU_WORKERS)) for i, task in enumerate(gpu_tasks)]
                futures.extend([executor.submit(process_image_gpu_task, task) for task in gpu_tasks_with_ids])
            
            # Submit CPU tasks
            if cpu_tasks:
                # Offset CPU worker IDs to avoid progress bar collision
                cpu_tasks_with_ids = [(*task, (i % num_cpu_workers) + NUM_GPU_WORKERS) for i, task in enumerate(cpu_tasks)]
                futures.extend([executor.submit(process_image_cpu_task, task) for task in cpu_tasks_with_ids])

            # Progress bar for images within the current dataset
            image_pbar = tqdm(total=len(tasks), desc=f"Images in {dataset_name}", unit="image", position=1, leave=False)
            for future in as_completed(futures):
                image_pbar.update(1)
                try:
                    metrics = future.result()
                    collected_metrics.append(metrics)
                except Exception as exc:
                    print(f'Task generated an exception: {exc}')

    # --- Save Metrics to CSV ---
    # Sort by dataset and image name for consistent output
    collected_metrics.sort(key=lambda m: (m['dataset'], m['image']))
    df = pd.DataFrame(collected_metrics)
    csv_output_path = os.path.join(metrics_dir, "fractal_cuda_parallel.csv")
    df.to_csv(csv_output_path, index=False)

    print(f"\nTest complete. Reconstructed images are saved in '{output_image_dir}'.")
    print(f"Metrics saved to '{csv_output_path}'.")

if __name__ == "__main__":
    # Set the CUDA_PATH environment variable to help child processes find it.
    _find_and_set_cuda_path()

    data_dirs = {
        "kodak": "data/kodak",
        "standard_test": "data/standard_test",
        "clic_subset": "data/clic_subset"
    }
    run_fractal_structured_test(data_dirs, "fractal_cuda_parallel.csv")