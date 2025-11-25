import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm


from src.wavelet_codec import wavelet_compress, wavelet_decompress, wavelet_save, wavelet_load
from src.fractal_codec import fractal_encode, fractal_decode, fractal_save, fractal_load

def estimate_bitrate(filesize_bytes, img_shape):
    # Rough bitrate: size in bits divided by number of pixels
    num_pixels = img_shape[0] * img_shape[1]
    return (filesize_bytes * 8) / num_pixels

def process_image(image_path, results_dir, dataset_name):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_name = os.path.basename(image_path)

    # Wavelet compression
    coeffs = wavelet_compress(img)
    wavelet_file = os.path.join(results_dir, f"{image_name}_wavelet.npz")
    wavelet_save(coeffs, wavelet_file)
    loaded_coeffs = wavelet_load(wavelet_file)
    img_wavelet = wavelet_decompress(loaded_coeffs)

    # Fractal compression
    fractal_params = fractal_encode(img_gray)
    fractal_file = os.path.join(results_dir, f"{image_name}_fractal.pkl")
    fractal_save(fractal_params, fractal_file)
    loaded_params = fractal_load(fractal_file)
    img_fractal = fractal_decode(loaded_params, img_gray.shape)

    # Save reconstructed images for inspection/comparison
    wavelet_recon_path = os.path.join(results_dir, "wavelet", dataset_name, f"{image_name}_wavelet.png")
    fractal_recon_path = os.path.join(results_dir, "fractal", dataset_name, f"{image_name}_fractal.png")

    os.makedirs(os.path.dirname(wavelet_recon_path), exist_ok=True)
    os.makedirs(os.path.dirname(fractal_recon_path), exist_ok=True)

    cv2.imwrite(wavelet_recon_path, img_wavelet)
    cv2.imwrite(fractal_recon_path, img_fractal)



    # Calculate metrics
    metrics = {}

    metrics['image'] = image_name

    metrics['psnr_wavelet'] = psnr(img_gray, img_wavelet)
    metrics['ssim_wavelet'] = ssim(img_gray, img_wavelet)

    metrics['psnr_fractal'] = psnr(img_gray, img_fractal)
    metrics['ssim_fractal'] = ssim(img_gray, img_fractal)

    # Bitrate estimates: filesize of saved compression files
    metrics['bitrate_wavelet'] = estimate_bitrate(os.path.getsize(wavelet_file), img_gray.shape)
    metrics['bitrate_fractal'] = estimate_bitrate(os.path.getsize(fractal_file), img_gray.shape)

    return metrics

def main(data_dirs, output_csv):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    collected_metrics = []

    for dataset_name, dataset_path in data_dirs.items():
        print(f"Processing dataset {dataset_name} from {dataset_path}")
        image_files = [fname for fname in os.listdir(dataset_path)
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        for fname in tqdm(image_files, desc=f"{dataset_name} images", unit="img"):
            image_path = os.path.join(dataset_path, fname)
            metrics = process_image(image_path, results_dir, dataset_name)
            metrics["dataset"] = dataset_name
            collected_metrics.append(metrics)

    df = pd.DataFrame(collected_metrics)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")

if __name__ == "__main__":
    data_dirs = {
        "kodak": "data/kodak",
        "standard_test": "data/standard_test",
        "clic_subset": "data/clic_subset"
    }
    main(data_dirs, "compression_comparison_metrics.csv")
