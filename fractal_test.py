import os
import cv2
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Make sure this script is run from the root of your project directory
# (i.e., the 'Project-Compression' folder)
from src.fractal_codec import fractal_encode, fractal_decode, fractal_save

def estimate_bitrate(filesize_bytes, img_shape):
    # Rough bitrate: size in bits divided by number of pixels
    num_pixels = img_shape[0] * img_shape[1]
    return (filesize_bytes * 8) / num_pixels

def run_fractal_test():
    """
    Tests the fractal encoding and decoding process on a subset of images
    and saves the reconstructed images.
    """
    # --- Configuration ---
    input_dir = "data/kodak"  # Directory containing test images
    output_dir = "results/test_fractal/"
    num_images_to_test = 5

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    try:
        all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not all_images:
            print(f"Error: No images found in '{input_dir}'. Please check the path.")
            return
        images_to_test = all_images[:num_images_to_test]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'. Please make sure the data is in the correct location.")
        return

    print(f"\nStarting fractal codec test on {len(images_to_test)} images from '{input_dir}'...")

    collected_metrics = []

    # --- Processing Loop ---
    for image_name in tqdm(images_to_test, desc="Processing Images", unit="image"):
        image_path = os.path.join(input_dir, image_name)

        # 1. Read the image in grayscale
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 2. Encode the image (with a detailed progress bar for this step)
        transformations = fractal_encode(img_gray, show_progress=True)

        # 3. Decode the image to reconstruct it
        reconstructed_img = fractal_decode(transformations, img_gray.shape)

        # 4. Save the reconstructed image
        output_path = os.path.join(output_dir, f"fractal_recon_{image_name}")
        cv2.imwrite(output_path, reconstructed_img)

        # 5. Calculate and store metrics
        fractal_file_path = os.path.join(output_dir, f"{image_name}_fractal_params.pkl")
        fractal_save(transformations, fractal_file_path)

        metrics = {
            'image': image_name,
            'psnr': psnr(img_gray, reconstructed_img),
            'ssim': ssim(img_gray, reconstructed_img),
            'bitrate': estimate_bitrate(os.path.getsize(fractal_file_path), img_gray.shape)
        }
        collected_metrics.append(metrics)

        # Clean up the temporary parameters file
        os.remove(fractal_file_path)

    # --- Save Metrics to CSV ---
    df = pd.DataFrame(collected_metrics)
    csv_output_path = os.path.join(output_dir, "test_metrics.csv")
    df.to_csv(csv_output_path, index=False)

    print(f"\nTest complete. Reconstructed images are saved in '{output_dir}'.")
    print(f"Metrics saved to '{csv_output_path}'.")

if __name__ == "__main__":
    run_fractal_test()