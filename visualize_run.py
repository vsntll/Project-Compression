import os
import sys
import argparse
import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add the 'src' directory to the Python path to allow importing codec modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from wavelet_codec import wavelet_load, wavelet_decompress, wavelet_compress
    from fractal_codec_structured import fractal_load, fractal_decompress, get_isometries, downsample_block
    from demo import visualize_wavelet_coeffs, visualize_fractal_matches
except ImportError as e:
    print(f"Error: Could not import a required module.")
    print("Please ensure 'demo.py' and the codec modules in 'src' are accessible.")
    print(f"Details: {e}")
    sys.exit(1)

def fractal_decompress_step_by_step(transformations, img_shape, range_size=8, domain_size=16, iterations=10):
    """
    A generator that yields the reconstructed image at each iteration of the fractal decoding process.
    """
    height, width = img_shape[:2]
    img = np.random.rand(height, width) # Start with random noise
    isometries = get_isometries()

    for i in range(iterations):
        new_img = np.zeros_like(img)
        counts = np.zeros_like(img, dtype=int)

        for (ri, rj, di, dj, a, b, iso_idx) in transformations:
            domain_block = img[di:di+domain_size, dj:dj+domain_size]
            domain_ds = downsample_block(domain_block, factor=domain_size//range_size)
            
            transformed_domain = isometries[iso_idx](domain_ds)
            block_pred = a * transformed_domain + b
            
            new_img[ri:ri+range_size, rj:rj+range_size] += block_pred
            counts[ri:ri+range_size, rj:rj+range_size] += 1

        counts[counts == 0] = 1
        img = new_img / counts

        # Yield the current state of the image, clipped and converted for visualization
        yield np.clip(img * 255, 0, 255).astype(np.uint8)

def wavelet_decompress_step_by_step(coeffs, output_shape, wavelet='haar'):
    """
    A generator that yields the reconstructed image at each level of the wavelet decoding process.
    Starts from the coarsest approximation and adds detail at each step.
    All intermediate frames are resized to the final output_shape.
    """
    import pywt
    # The coefficients are ordered from coarsest (cA) to finest (details)
    # e.g., [cA_level, (cH_level, cV_level, cD_level), ..., (cH_1, cV_1, cD_1)]
    
    target_size = (output_shape[1], output_shape[0]) # (width, height) for cv2.resize

    # Start with the approximation coefficients of the highest level
    current_recon = coeffs[0]
    
    # Yield the initial low-frequency approximation
    frame = np.clip(pywt.waverec2([current_recon], wavelet), 0, 255).astype(np.uint8)
    yield cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)

    # Iteratively apply the inverse transform for each level of detail
    for i in range(1, len(coeffs)):
        # Combine the current approximation with all detail levels up to this point
        coeffs_to_reconstruct = coeffs[:i+1]
        current_recon = pywt.waverec2(coeffs_to_reconstruct, wavelet)
        frame = np.clip(current_recon, 0, 255).astype(np.uint8)
        yield cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)

def main(run_dir):
    """
    Loads data from a completed demo run and generates visualizations.
    """
    print(f"--- Visualizing Run from Directory: {run_dir} ---")

    # --- 1. Identify files ---
    run_name = os.path.basename(run_dir)
    image_name = run_name.split('_c')[0]
    
    original_image_path = f"data/kodak/{image_name}.png"
    wavelet_compressed_path = os.path.join(run_dir, f"{image_name}_wavelet_compressed.npz")
    fractal_compressed_path = os.path.join(run_dir, f"{image_name}_fractal_compressed.pkl")
    wavelet_recon_path = os.path.join(run_dir, f"{image_name}_reconstructed_wavelet.png")
    fractal_recon_path = os.path.join(run_dir, f"{image_name}_reconstructed_fractal.png")

    # Check if essential files exist
    for f in [original_image_path, wavelet_compressed_path, fractal_compressed_path, wavelet_recon_path, fractal_recon_path]:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            print("Please ensure you have run 'demo.py' first and are pointing to a valid run directory.")
            return

    # --- 2. Load all data ---
    print("Loading images and compressed data...")
    original_image_color = iio.imread(original_image_path)
    if original_image_color.ndim == 3:
        original_image_gray = np.dot(original_image_color[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        original_image_gray = original_image_color

    wavelet_reconstructed = iio.imread(wavelet_recon_path)
    fractal_reconstructed = iio.imread(fractal_recon_path)
    fractal_params = fractal_load(fractal_compressed_path)
    wavelet_loaded_coeffs = wavelet_load(wavelet_compressed_path)
    # Re-generate wavelet coeffs for visualization (they are not saved in the demo)
    wavelet_coeffs = wavelet_compress(original_image_color)

    # --- 3. Generate Fractal Decompression GIF ---
    print("Generating fractal decompression steps visualization (GIF)...")
    fractal_steps_path = os.path.join(run_dir, f"{image_name}_fractal_decompression_steps.gif")
    
    # Use the step-by-step decoder
    iterations = 12
    frames = list(fractal_decompress_step_by_step(fractal_params, original_image_gray.shape, iterations=iterations))
    
    # Add a title to each frame
    titled_frames = []
    for i, frame in enumerate(frames):
        # Convert to color to draw text
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        title = f"Fractal Iteration: {i + 1}/{iterations}"
        cv2.putText(frame_color, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        titled_frames.append(frame_color)

    iio.mimsave(fractal_steps_path, titled_frames, duration=3.0, loop=0)
    print(f"Saved fractal steps GIF to: {fractal_steps_path}")

    # --- 4. Generate Wavelet Decompression GIF ---
    print("Generating wavelet decompression steps visualization (GIF)...")
    wavelet_steps_path = os.path.join(run_dir, f"{image_name}_wavelet_decompression_steps.gif")

    # Use the step-by-step wavelet decoder
    # We need to dequantize the coefficients first before reconstruction
    from wavelet_codec import wavelet_decompress
    wavelet_full_recon = wavelet_decompress(wavelet_loaded_coeffs) # Use this to get quant_step if needed, or assume

    frames = list(wavelet_decompress_step_by_step(wavelet_loaded_coeffs, original_image_gray.shape))
    
    titled_frames = []
    for i, frame in enumerate(frames):
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        title = f"Wavelet Level: {i + 1}/{len(frames)}"
        cv2.putText(frame_color, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        titled_frames.append(frame_color)

    iio.mimsave(wavelet_steps_path, titled_frames, duration=3.0, loop=0)
    print(f"Saved wavelet steps GIF to: {wavelet_steps_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from a completed compression run.")
    parser.add_argument("run_directory", type=str, help="Path to the specific run directory inside 'results/demo/'.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_directory):
        print(f"Error: Directory not found at '{args.run_directory}'")
        sys.exit(1)
        
    main(args.run_directory)
