import os
import cv2
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Make sure this script is run from the root of your project directory
from src.wavelet_codec import wavelet_compress, wavelet_decompress, wavelet_save, wavelet_load

def estimate_bitrate(filesize_bytes, img_shape):
    # Rough bitrate: size in bits divided by number of pixels
    num_pixels = img_shape[0] * img_shape[1]
    return (filesize_bytes * 8) / num_pixels

def run_wavelet_test():
    """
    Tests the wavelet encoding and decoding process on a subset of images
    and saves the reconstructed images and metrics.
    """
    # --- Configuration ---
    input_dir = "data/kodak"  # Directory containing test images
    output_dir = "results/test_wavelet/"
    metrics_dir = "results/metrics/"
    num_images_to_test = 5

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_dir}")
    print(f"Metrics will be saved to: {metrics_dir}")

    try:
        all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not all_images:
            print(f"Error: No images found in '{input_dir}'. Please check the path.")
            return
        images_to_test = all_images[:num_images_to_test]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'. Please make sure the data is in the correct location.")
        return

    print(f"\nStarting wavelet codec test on {len(images_to_test)} images from '{input_dir}'...")

    collected_metrics = []

    # --- Processing Loop ---
    for image_name in tqdm(images_to_test, desc="Processing Images", unit="image"):
        image_path = os.path.join(input_dir, image_name)

        # 1. Read the image
        img_color = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 2. Compress the image
        coeffs_quant = wavelet_compress(img_color)

        # 3. Save and load the compressed data to measure file size
        wavelet_file_path = os.path.join(output_dir, f"{image_name}_wavelet.npz")
        wavelet_save(coeffs_quant, wavelet_file_path)
        loaded_coeffs = wavelet_load(wavelet_file_path)

        # 4. Decompress the image to reconstruct it
        reconstructed_img = wavelet_decompress(loaded_coeffs)

        # 5. Save the reconstructed image
        output_path = os.path.join(output_dir, f"wavelet_recon_{image_name}")
        cv2.imwrite(output_path, reconstructed_img)

        # 6. Calculate and store metrics
        metrics = {
            'image': image_name,
            'psnr': psnr(img_gray, reconstructed_img),
            'ssim': ssim(img_gray, reconstructed_img),
            'bitrate': estimate_bitrate(os.path.getsize(wavelet_file_path), img_gray.shape)
        }
        collected_metrics.append(metrics)

    # --- Save Metrics to CSV ---
    df = pd.DataFrame(collected_metrics)
    csv_output_path = os.path.join(metrics_dir, "wavelet_test_metrics.csv")
    df.to_csv(csv_output_path, index=False)

    print(f"\nTest complete. Reconstructed images are saved in '{output_dir}'.")
    print(f"Metrics saved to '{csv_output_path}'.")

if __name__ == "__main__":
    run_wavelet_test()