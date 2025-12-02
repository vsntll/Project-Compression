import os
import time
import imageio.v2 as iio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from skimage.metrics import structural_similarity as ssim

# Add the 'src' directory to the Python path to allow importing codec modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from wavelet_codec import wavelet_compress, wavelet_decompress, wavelet_save, wavelet_load
    from fractal_codec_structured import fractal_compress, fractal_decompress, fractal_save, fractal_load
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

def main():
    """
    A demo script to run and compare wavelet and fractal compression on a single image.
    """
    # --- Configuration ---
    # Use an image from the Kodak dataset. Change this path to use a different image.
    image_path = "data/kodak/kodim13.png"
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "results/demo"

    # --- Compression Control ---
    # A value from 1-100. Higher values mean more compression (lower quality).
    compression_level = 25  # e.g., 10=low compression, 50=medium, 80=high

    # --- Fractal Quality Preset ---
    # 'fast': Prioritizes speed with higher distortion tolerance.
    # 'medium': A balance between speed and quality.
    # 'quality': Prioritizes quality with very low distortion tolerance (much slower).
    fractal_quality_preset = 'medium'

    # Create output directory if it doesn't exist
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
    print("\n--- Starting Fractal Compression (CPU) ---")
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

    fractal_params = fractal_compress(original_image_color, show_progress=True, distortion_threshold=distortion_threshold)
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

    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    main()