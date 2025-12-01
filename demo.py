import os
import time
import imageio.v2 as iio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
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
    image_path = "data/kodak/kodim01.png"
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "results/demo"

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
    # The codec expects a color image to convert to gray, so we read it again.
    wavelet_coeffs = wavelet_compress(original_image_color)
    wavelet_save(wavelet_coeffs, wavelet_compressed_path)
    wavelet_compress_time = time.time() - start_time
    
    start_time = time.time()
    wavelet_reconstructed = wavelet_decompress(wavelet_load(wavelet_compressed_path))
    wavelet_decompress_time = time.time() - start_time

    # Save reconstructed image
    wavelet_output_path = os.path.join(output_dir, f"{image_name}_reconstructed_wavelet.png")
    iio.imwrite(wavelet_output_path, wavelet_reconstructed)

    # Calculate metrics
    psnr_w = psnr(original_image_gray, wavelet_reconstructed)
    ssim_w = ssim(original_image_gray, wavelet_reconstructed, data_range=255)
    bitrate_w = get_bitrate(wavelet_compressed_path, original_image_gray)

    print(f"Wavelet compression time: {wavelet_compress_time:.4f}s")
    print(f"Wavelet decompression time: {wavelet_decompress_time:.4f}s")
    print(f"Reconstructed image saved to: {wavelet_output_path}")
    print(f"Metrics (Wavelet) -> PSNR: {psnr_w:.2f} dB, SSIM: {ssim_w:.4f}, Bitrate: {bitrate_w:.4f} bpp")

    # --- 2. Fractal Compression (CPU) ---
    print("\n--- Starting Fractal Compression (CPU) ---")
    print("(This may take several minutes depending on your CPU and image size)")
    fractal_compressed_path = os.path.join(output_dir, f"{image_name}_fractal_compressed.pkl")

    start_time = time.time()
    fractal_params = fractal_compress(original_image_color, show_progress=True)
    fractal_save(fractal_params, fractal_compressed_path)
    fractal_compress_time = time.time() - start_time

    start_time = time.time()
    loaded_params = fractal_load(fractal_compressed_path)
    fractal_reconstructed = fractal_decompress(loaded_params, original_image_gray.shape)
    fractal_decompress_time = time.time() - start_time

    # Save reconstructed image
    fractal_output_path = os.path.join(output_dir, f"{image_name}_reconstructed_fractal.png")
    iio.imwrite(fractal_output_path, fractal_reconstructed)

    # Calculate metrics
    psnr_f = psnr(original_image_gray, fractal_reconstructed)
    ssim_f = ssim(original_image_gray, fractal_reconstructed, data_range=255)
    bitrate_f = get_bitrate(fractal_compressed_path, original_image_gray)

    print(f"Fractal compression time: {fractal_compress_time:.4f}s")
    print(f"Fractal decompression time: {fractal_decompress_time:.4f}s")
    print(f"Reconstructed image saved to: {fractal_output_path}")
    print(f"Metrics (Fractal) -> PSNR: {psnr_f:.2f} dB, SSIM: {ssim_f:.4f}, Bitrate: {bitrate_f:.4f} bpp")

    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    main()