import os
import cv2
import numpy as np
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

def process_chunk_gpu_task(args):
    """
    A self-contained function to process a single image chunk on the GPU.
    """
    image_path, dataset_name, dataset_output_dir, chunk_index, num_chunks, worker_id = args
    worker_id += 2 # Offset for dataset and image progress bars
    image_name = os.path.basename(image_path)
    chunk_name = f"{os.path.splitext(image_name)[0]}_chunk{chunk_index}"

    # 1. Read the image
    img_color = cv2.imread(image_path)
    
    # Determine the chunk (horizontal strip) for this task
    h, w, _ = img_color.shape
    chunk_height = h // num_chunks
    start_row = chunk_index * chunk_height
    end_row = (chunk_index + 1) * chunk_height if chunk_index < num_chunks - 1 else h
    img_chunk_color = img_color[start_row:end_row, :]

    # 2. Compress the image, passing the worker_id as the position for the progress bar
    compressed_data = fractal_compress_cuda(img_chunk_color, show_progress=True, position=worker_id, distortion_threshold=0.001)

    # Return the compressed data and original chunk shape for later reassembly
    return {
        'image_name': image_name,
        'dataset_name': dataset_name,
        'chunk_index': chunk_index,
        'compressed_data': compressed_data,
        'original_shape': img_chunk_color.shape,
        'type': 'gpu'
    }

def process_chunk_cpu_task(args):
    """
    A self-contained function to process a single image chunk on the CPU.
    """
    image_path, dataset_name, dataset_output_dir, chunk_index, num_chunks, worker_id = args
    worker_id += 2 # Offset for dataset and image progress bars
    image_name = os.path.basename(image_path)
    chunk_name = f"{os.path.splitext(image_name)[0]}_chunk{chunk_index}"

    # 1. Read the image
    img_color = cv2.imread(image_path)
    
    # Determine the chunk (horizontal strip) for this task
    h, w, _ = img_color.shape
    chunk_height = h // num_chunks
    start_row = chunk_index * chunk_height
    end_row = (chunk_index + 1) * chunk_height if chunk_index < num_chunks - 1 else h
    img_chunk_color = img_color[start_row:end_row, :]

    # 2. Compress the image using the CPU codec
    compressed_data = fractal_compress_cpu(img_chunk_color, show_progress=True, position=worker_id, distortion_threshold=0.005)

    # Return the compressed data and original chunk shape for later reassembly
    return {
        'image_name': image_name,
        'dataset_name': dataset_name,
        'chunk_index': chunk_index,
        'compressed_data': compressed_data,
        'original_shape': img_chunk_color.shape,
        'type': 'cpu'
    }

def reassemble_and_calculate_metrics(image_name, dataset_name, dataset_output_dir, chunk_results, num_chunks):
    """
    Reassembles chunks, saves the final image, and calculates metrics.
    """
    # Sort chunks by index to ensure correct order
    chunk_results.sort(key=lambda r: r['chunk_index'])

    # 1. Reassemble the image
    reconstructed_chunks = []
    total_file_size = 0
    
    for result in chunk_results:
        if result['type'] == 'gpu':
            reconstructed_chunks.append(fractal_decompress_cuda(result['compressed_data']))
        else: # cpu
            reconstructed_chunks.append(fractal_decompress_cpu(result['compressed_data']))

        # Save each chunk's compressed data to a temporary file to measure its size
        temp_path = os.path.join(dataset_output_dir, f"{os.path.splitext(image_name)[0]}_chunk_{result['chunk_index']}.pkl")
        fractal_save(result['compressed_data'], temp_path)
        total_file_size += os.path.getsize(temp_path)
        os.remove(temp_path) # Clean up temporary file

    reconstructed_img = np.vstack(reconstructed_chunks)

    # 2. Save the fully reconstructed image
    output_path = os.path.join(dataset_output_dir, f"fractal_recon_{image_name}")
    cv2.imwrite(output_path, reconstructed_img)

    # 3. Load original image for metric calculation
    original_image_path = os.path.join("data", dataset_name, image_name)
    img_color = cv2.imread(original_image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 4. Calculate and return final metrics for the whole image
    metrics = {
        'dataset': dataset_name,
        'image': image_name,
        'psnr': psnr(img_gray, reconstructed_img, data_range=255),
        'ssim': ssim(img_gray, reconstructed_img, data_range=255),
        'bitrate': estimate_bitrate(total_file_size, img_gray.shape)
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
    NUM_CHUNKS_PER_IMAGE = 4 # Configurable number of chunks to split each image into

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
    NUM_GPU_WORKERS = min(4, max_workers - 1 if max_workers > 1 else 1) # Ensure at least 1 CPU worker if possible
    num_cpu_workers = max_workers - NUM_GPU_WORKERS

    print(f"Allocating {NUM_GPU_WORKERS} worker(s) for GPU tasks and {num_cpu_workers} for CPU tasks to keep GPU utilization below 100%.")
    # Outer loop for datasets, with its own progress bar
    dataset_pbar = tqdm(data_dirs.items(), desc="Total Datasets", unit="dataset", position=0)
    for dataset_name, dataset_path in dataset_pbar:
        dataset_pbar.set_description(f"Processing Dataset: {dataset_name}")

        tasks = []
        # --- Prepare chunk-based tasks for the current dataset ---
        try: 
            image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            if not image_files: 
                print(f"Warning: No images found in '{dataset_path}'. Skipping.")
                continue

            dataset_output_dir = os.path.join(output_image_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            for image_name in image_files:
                for i in range(NUM_CHUNKS_PER_IMAGE):
                    image_path = os.path.join(dataset_path, image_name)
                    tasks.append((image_path, dataset_name, dataset_output_dir, i, NUM_CHUNKS_PER_IMAGE))

        except FileNotFoundError:
            print(f"Error: Input directory not found at '{dataset_path}'. Skipping.")
            continue

        # --- Sort tasks by size to prioritize larger images for GPU workers ---
        print(f"Sorting {len(tasks)} tasks for {dataset_name} by image size...")
        tasks.sort(key=lambda task: os.path.getsize(task[0]), reverse=True)

        completed_chunks_by_image = {}

        # --- Dynamic Task Execution for the current dataset ---
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create iterators for task arguments and available worker IDs
            task_iterator = iter(tasks)
            
            # Map to track which future is running on which worker (and its type)
            # Values will be tuples: (worker_id, 'gpu') or (worker_id, 'cpu')
            future_to_worker = {}
            # New map to track the task arguments for each future, enabling task re-submission
            future_to_task_args = {}
            # Keep a reference to CPU futures to check for potential tasks to steal
            cpu_futures = set()

            # --- Initial task submission to fill all workers ---
            # Prioritize GPU workers first
            for worker_id in range(NUM_GPU_WORKERS):
                try:
                    task_args = next(task_iterator)
                    future = executor.submit(process_chunk_gpu_task, (*task_args, worker_id))
                    future_to_task_args[future] = task_args
                    future_to_worker[future] = (worker_id, 'gpu')
                except StopIteration:
                    break # No more tasks
            
            # Then fill CPU workers
            for worker_id in range(num_cpu_workers):
                try:
                    task_args = next(task_iterator)
                    # Offset CPU worker IDs to avoid progress bar collision
                    future = executor.submit(process_chunk_cpu_task, (*task_args, worker_id + NUM_GPU_WORKERS))
                    future_to_task_args[future] = task_args
                    cpu_futures.add(future)
                    future_to_worker[future] = (worker_id, 'cpu')
                except StopIteration:
                    break # No more tasks

            # Progress bar for images within the current dataset
            image_pbar = tqdm(total=len(tasks), desc=f"Images in {dataset_name}", unit="image", position=1, leave=False)

            # --- Process tasks as they complete and submit new ones ---
            # as_completed() works on a snapshot. We need a loop that continues as long
            # as there are active futures.
            while future_to_worker:
                # Wait for the next future to complete
                done_future = next(as_completed(future_to_worker.keys()))

                image_pbar.update(1)
                try:
                    chunk_result = done_future.result()
                    image_name = chunk_result['image_name']

                    if image_name not in completed_chunks_by_image:
                        completed_chunks_by_image[image_name] = []
                    
                    completed_chunks_by_image[image_name].append(chunk_result)

                    # If all chunks for an image are done, reassemble and calculate metrics
                    if len(completed_chunks_by_image[image_name]) == NUM_CHUNKS_PER_IMAGE:
                        final_metrics = reassemble_and_calculate_metrics(image_name, dataset_name, dataset_output_dir, completed_chunks_by_image[image_name], NUM_CHUNKS_PER_IMAGE)
                        collected_metrics.append(final_metrics)
                        del completed_chunks_by_image[image_name]

                except Exception as exc:
                    print(f'Task generated an exception: {exc}')

                # A worker has finished. Get its ID and type.
                # Also remove the completed task from our tracking maps.
                worker_id, worker_type = future_to_worker.pop(done_future)
                future_to_task_args.pop(done_future)
                if done_future in cpu_futures:
                    cpu_futures.remove(done_future)

                # Try to submit a new task to the now-free worker
                try:
                    # --- STAGE 1: Get a new task from the main iterator ---
                    task_args = next(task_iterator)
                    if worker_type == 'gpu':
                        new_future = executor.submit(process_chunk_gpu_task, (*task_args, worker_id))
                    else: # worker_type == 'cpu'
                        new_future = executor.submit(process_chunk_cpu_task, (*task_args, worker_id + NUM_GPU_WORKERS))
                        cpu_futures.add(new_future)
                    # Add the new task to our tracking dictionary
                    future_to_task_args[new_future] = task_args
                    future_to_worker[new_future] = (worker_id, worker_type)
                except StopIteration:
                    # --- STAGE 2: No new tasks. A GPU can now steal from a CPU. ---
                    if worker_type == 'gpu' and cpu_futures:
                        # Find a CPU task to steal
                        cpu_future_to_steal = cpu_futures.pop()
                        
                        # Cancel the CPU task. This may not stop it if it's already running, but it prevents it from starting.
                        cpu_future_to_steal.cancel()
                        stolen_task_args = future_to_task_args.pop(cpu_future_to_steal)
                        future_to_worker.pop(cpu_future_to_steal)
                        print(f"\nGPU worker {worker_id} is stealing task for image {os.path.basename(stolen_task_args[0])} from a CPU worker.")
                        # Submit the stolen task to the GPU
                        new_future = executor.submit(process_chunk_gpu_task, (*stolen_task_args, worker_id))
                        future_to_worker[new_future] = (worker_id, 'gpu')
                        future_to_task_args[new_future] = stolen_task_args

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