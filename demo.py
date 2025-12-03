import os
import time
import imageio.v2 as iio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import csv
import cv2
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
# Add the 'src' directory to the Python path to allow importing codec modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from wavelet_codec import wavelet_compress, wavelet_decompress, wavelet_save, wavelet_load
    from fractal_codec_structured import fractal_compress_parallel, fractal_decompress, fractal_save, fractal_load
except ImportError as e:
    print(f"Error: Could not import codec modules.")
    print("Please ensure that 'src/wavelet_codec.py' and 'src/fractal_codec_structured.py' exist and have the correct function names.")
    print(f"Details: {e}")
    sys.exit(1)

def get_bitrate(filepath, original_image):
    """Calculates the bitrate in bits per pixel."""
    compressed_size_bytes = os.path.getsize(filepath)
    num_pixels = original_image.shape[0] * original_image.shape[1]
    return (compressed_size_bytes * 8) / num_pixels

def visualize_wavelet_coeffs(coeffs, level):
    """Creates a single image visualizing the wavelet coefficients."""
    # Normalize coefficient arrays for visualization
    processed_coeffs = []
    for arr in coeffs:
        if isinstance(arr, tuple):
            # Detail coefficients
            processed_coeffs.append(tuple(cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for c in arr))
        else:
            # Approximation coefficients
            processed_coeffs.append(cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    # Arrange coefficients into a single image
    cA = processed_coeffs[0]
    rows_cA, cols_cA = cA.shape

    # Calculate the full canvas size required by summing the dimensions of the sub-bands
    full_height = rows_cA + processed_coeffs[1][0].shape[0]
    full_width = cols_cA + processed_coeffs[1][0].shape[1]
    
    canvas = np.zeros((full_height, full_width), dtype=np.uint8)
    canvas[:rows_cA, :cols_cA] = cA

    for i in range(level):
        (cH, cV, cD) = processed_coeffs[i + 1]
        rows_cH, cols_cH = cH.shape
        canvas[:rows_cH, cols_cA:cols_cA + cols_cH] = cH
        canvas[rows_cA:rows_cA + rows_cH, :cols_cH] = cV
        canvas[rows_cA:rows_cA + rows_cH, cols_cA:cols_cA + cols_cH] = cD

    return canvas

def visualize_fractal_matches(transformations, img_shape):
    """Creates an image visualizing which domain block maps to each range block."""
    vis_image = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    range_size = transformations[0][0] - transformations[1][0] if len(transformations) > 1 else 8 # Infer range size
    range_size = abs(range_size) if range_size != 0 else 8

    for (ri, rj, di, dj, a, b, iso_idx) in transformations:
        color = ((di + dj) % 255, (di * 2) % 255, (dj * 2) % 255) # Map domain location to a color
        cv2.rectangle(vis_image, (rj, ri), (rj + range_size, ri + range_size), color, -1)
    return vis_image

def save_results_to_csv(filepath, data):
    """Appends a dictionary of results to a CSV file."""
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data)

def get_system_info():
    """Gathers basic system information for logging."""
    import platform
    try:
        # Use os.cpu_count() for modern Python versions
        cpu_cores = os.cpu_count()
    except AttributeError:
        # Fallback for older Python
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count()
    return f"{platform.system()} {platform.release()}; CPU Cores: {cpu_cores}"

def main():
    """
    A demo script to run and compare wavelet and fractal compression on a single image.
    """
    # --- Configuration ---
    # Use an image from the Kodak dataset. Change this path to use a different image.
    image_path = "data/kodak/kodim05.png"
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # --- Compression Control ---
    # A value from 1-100. Higher values mean more compression (lower quality).
    compression_level = 25  # e.g., 10=low compression, 50=medium, 80=high

    # --- Fractal Quality Preset ---
    # 'fast': Prioritizes speed with higher distortion tolerance.
    # 'medium': A balance between speed and quality.
    # 'quality': Prioritizes quality with very low distortion tolerance (much slower).
    fractal_quality_preset = 'fast'

    # Create a unique output directory for this run to avoid overwriting results
    run_name = f"{image_name}_c{compression_level}_{fractal_quality_preset}"
    output_dir = os.path.join("results", "demo", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Original Image ---
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        print("Please download the standard test images and place them in 'data/standard_test/'.")
        return

    print(f"Loading image: {image_path}")
    original_image_color = iio.imread(image_path)
    original_image_gray = original_image_color # Will be overwritten if color
    # Convert to grayscale for compatibility with the codecs
    if original_image_color.ndim == 3:
        original_image_gray = np.dot(original_image_color[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # --- 1. Wavelet Compression ---
    print("\n--- Starting Wavelet Compression ---")
    wavelet_compressed_path = os.path.join(output_dir, f"{image_name}_wavelet_compressed.npz")
    
    start_time = time.time()
    # We introduce a 'quantization_step' parameter to control wavelet compression.
    # This is derived from the compression_level.
    # The formula can be adjusted, but here we map 1-100 to a reasonable range.
    quantization_step = int(compression_level * 0.5) + 1 # Map 1-100 -> 1-51
    wavelet_coeffs = wavelet_compress(original_image_color, quant_step=quantization_step)
    wavelet_save(wavelet_coeffs, wavelet_compressed_path)
    wavelet_compress_time = time.time() - start_time
    
    start_time = time.time()
    wavelet_reconstructed = wavelet_decompress(wavelet_load(wavelet_compressed_path), quant_step=quantization_step)
    wavelet_decompress_time = time.time() - start_time

    # Ensure the reconstructed image has the same dimensions as the original
    if wavelet_reconstructed.shape != original_image_gray.shape:
        print(f"Resizing wavelet image from {wavelet_reconstructed.shape} to {original_image_gray.shape}")
        wavelet_reconstructed = cv2.resize(wavelet_reconstructed, (original_image_gray.shape[1], original_image_gray.shape[0]))

    # Save reconstructed image
    wavelet_output_path = os.path.join(output_dir, f"{image_name}_reconstructed_wavelet.png")
    iio.imwrite(wavelet_output_path, wavelet_reconstructed)

    # Calculate metrics
    psnr_w = psnr(original_image_gray, wavelet_reconstructed)
    ssim_w = ssim(original_image_gray, wavelet_reconstructed, data_range=255)
    bitrate_w = get_bitrate(wavelet_compressed_path, original_image_gray)

    print(f"Using Wavelet Quantization Step: {quantization_step}")
    print(f"Wavelet compression time: {wavelet_compress_time:.4f}s")
    print(f"Wavelet decompression time: {wavelet_decompress_time:.4f}s")
    print(f"Reconstructed image saved to: {wavelet_output_path}")
    print(f"Metrics (Wavelet) -> Size: {wavelet_reconstructed.shape}, PSNR: {psnr_w:.2f} dB, SSIM: {ssim_w:.4f}, Bitrate: {bitrate_w:.4f} bpp")

    # --- 2. Fractal Compression (CPU) ---
    print("\n--- Starting Fractal Compression ---")
    print("(This may take several minutes depending on your CPU and image size)")
    fractal_compressed_path = os.path.join(output_dir, f"{image_name}_fractal_compressed.pkl")

    start_time = time.time()
    # We use a 'distortion_threshold' to control fractal compression.
    # A higher threshold allows more distortion, leading to more compression.
    # We derive this from the compression_level and quality preset.
    if fractal_quality_preset == 'quality':
        # Use a very small threshold to force the search for better matches.
        distortion_threshold = 0.001
    elif fractal_quality_preset == 'medium':
        # An intermediate threshold for a balance of speed and quality.
        distortion_threshold = (compression_level / 2000.0) + 0.002 # Map 1-100 -> 0.002 to 0.052
    else: # 'fast' is the default
        distortion_threshold = (compression_level / 1000.0) + 0.001 # Map 1-100 -> 0.002 to 0.101

    fractal_params = fractal_compress_parallel(original_image_color, show_progress=True, distortion_threshold=distortion_threshold)
    fractal_save(fractal_params, fractal_compressed_path)
    fractal_compress_time = time.time() - start_time

    start_time = time.time()
    loaded_params = fractal_load(fractal_compressed_path)
    fractal_reconstructed = fractal_decompress(loaded_params, original_image_gray.shape)
    fractal_decompress_time = time.time() - start_time

    # Ensure the reconstructed image has the same dimensions as the original
    if fractal_reconstructed.shape != original_image_gray.shape:
        print(f"Resizing fractal image from {fractal_reconstructed.shape} to {original_image_gray.shape}")
        fractal_reconstructed = cv2.resize(fractal_reconstructed, (original_image_gray.shape[1], original_image_gray.shape[0]))

    # Save reconstructed image
    fractal_output_path = os.path.join(output_dir, f"{image_name}_reconstructed_fractal.png")
    iio.imwrite(fractal_output_path, fractal_reconstructed)

    # Calculate metrics
    psnr_f = psnr(original_image_gray, fractal_reconstructed)
    ssim_f = ssim(original_image_gray, fractal_reconstructed, data_range=255)
    bitrate_f = get_bitrate(fractal_compressed_path, original_image_gray)

    print(f"Using Fractal Distortion Threshold: {distortion_threshold:.4f}")
    print(f"Fractal compression time: {fractal_compress_time:.4f}s")
    print(f"Fractal decompression time: {fractal_decompress_time:.4f}s")
    print(f"Reconstructed image saved to: {fractal_output_path}")
    print(f"Metrics (Fractal) -> Size: {fractal_reconstructed.shape}, PSNR: {psnr_f:.2f} dB, SSIM: {ssim_f:.4f}, Bitrate: {bitrate_f:.4f} bpp")

   
    # --- 4. Save results to CSV ---
    csv_path = "results/compression_results.csv"
    print(f"\nSaving results to {csv_path}...")
    
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image": image_name,
        "image_dimensions": f"{original_image_gray.shape[1]}x{original_image_gray.shape[0]}",
        "compression_level": compression_level,
        "fractal_preset": fractal_quality_preset,
        "wavelet_psnr_db": f"{psnr_w:.2f}",
        "wavelet_ssim": f"{ssim_w:.4f}",
        "wavelet_bitrate_bpp": f"{bitrate_w:.4f}",
        "wavelet_comp_time_s": f"{wavelet_compress_time:.4f}",
        "fractal_psnr_db": f"{psnr_f:.2f}",
        "fractal_ssim": f"{ssim_f:.4f}",
        "fractal_bitrate_bpp": f"{bitrate_f:.4f}",
        "fractal_comp_time_s": f"{fractal_compress_time:.4f}",
        "system_info": get_system_info()
    }
    
    save_results_to_csv(csv_path, results_data)
    print("Results saved.")

if __name__ == "__main__":
    main()