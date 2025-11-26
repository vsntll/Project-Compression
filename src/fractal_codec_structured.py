import numpy as np
import cv2
import pickle
from tqdm import tqdm

# --- Core Fractal Functions (self-contained) ---

def _downsample_block(block, factor=2):
    """Downsamples a block by an integer factor."""
    return block.reshape((block.shape[0]//factor, factor, block.shape[1]//factor, factor)).mean(axis=(1,3))

def _fractal_encode_internal(img, range_size, domain_size, show_progress, distortion_threshold, max_dfs_visits, position=0):
    """Internal fractal encoding logic."""
    height, width = img.shape
    img_float = img.astype(np.float32) / 255.0
    transformations = []

    # --- DFS Optimization: Create a grid of domain blocks for neighbor lookups ---
    domain_stride = range_size // 2
    domain_grid_h = (height - domain_size) // domain_stride + 1
    domain_grid_w = (width - domain_size) // domain_stride + 1
    domain_blocks = np.empty((domain_grid_h, domain_grid_w), dtype=object)
    domain_block_cache = {} # Cache for downsampled blocks and their stats

    for i in range(domain_grid_h):
        for j in range(domain_grid_w):
            di, dj = i * domain_stride, j * domain_stride
            domain_block = img_float[di:di+domain_size, dj:dj+domain_size]
            domain_blocks[i, j] = (di, dj) # Store coordinates only

            # Pre-calculate and cache downsampled blocks and their stats
            domain_ds = _downsample_block(domain_block, factor=domain_size // range_size)
            var_x = np.var(domain_ds)
            mean_x = np.mean(domain_ds)
            domain_block_cache[(i, j)] = (domain_ds, var_x, mean_x)

    range_blocks_coords = [(i, j) for i in range(0, height - range_size + 1, range_size) for j in range(0, width - range_size + 1, range_size)]
    
    pbar = tqdm(range_blocks_coords, desc=f"CPU Worker {position}", unit="block", disable=not show_progress, leave=False, position=position)
    for ri, rj in pbar:
        range_block = img_float[ri:ri+range_size, rj:rj+range_size]
        min_err = float('inf')
        best_params_for_block = None

        # Pre-calculate stats for the range block
        mean_y = np.mean(range_block)
        y_flat = range_block.flatten()

        # --- Exhaustive Search (replaces DFS) ---
        # Iterate through all domain blocks instead of a limited search
        for curr_i in range(domain_grid_h):
            for curr_j in range(domain_grid_w):
                # --- Perform the comparison ---
                di, dj = domain_blocks[curr_i, curr_j]
                domain_ds, var_x, mean_x = domain_block_cache[(curr_i, curr_j)]

                if var_x > 1e-6:
                    # Efficiently calculate covariance term: cov(x,y) = E[xy] - E[x]E[y]
                    cov_xy = np.mean(domain_ds.flatten() * y_flat) - (mean_x * mean_y)
                    a = cov_xy / var_x
                    b = mean_y - a * mean_x
                    pred = a * domain_ds + b
                    err = np.mean((range_block - pred)**2)
        
                    if err < min_err:
                        min_err = err
                        best_params_for_block = (ri, rj, di, dj, a, b)
        
                # Early exit if a good enough match is found
                if min_err < distortion_threshold:
                    break
            if min_err < distortion_threshold: # This break is for the outer loop
                 break

        if best_params_for_block:
            transformations.append(best_params_for_block)
    return transformations


def _fractal_decode_internal(transformations, img_shape, range_size, domain_size, iterations):
    """Internal fractal decoding logic."""
    height, width = img_shape
    img = np.zeros(img_shape, dtype=np.float32)
    for _ in range(iterations):
        new_img = np.zeros_like(img)
        for (ri, rj, di, dj, a, b) in transformations:
            domain_block = img[di:di+domain_size, dj:dj+domain_size]
            domain_ds = _downsample_block(domain_block, factor=domain_size // range_size)
            block_pred = np.clip(a * domain_ds + b, 0, 1)
            new_img[ri:ri+range_size, rj:rj+range_size] = block_pred
        img = new_img
    return (img * 255).astype(np.uint8)

# --- Public API Functions ---
def fractal_compress(img, range_size=8, domain_size=16, show_progress=False, distortion_threshold=0.005, max_dfs_visits=500, position=0):
    """
    Compresses a grayscale image using fractal encoding.
    This is a wrapper around the original fractal_encode function.
    """
    # Fractal encoding works on grayscale images
    if len(img.shape) > 2 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # The 'compressed' data is the list of transformation parameters
    transformations = _fractal_encode_internal(img_gray, range_size, domain_size, show_progress, distortion_threshold, max_dfs_visits, position)
    
    # We also need the original image shape for decoding
    return transformations, img_gray.shape

def fractal_decompress(compressed_data, iterations=10):
    """
    Decompresses fractal-encoded data to reconstruct an image.
    This is a wrapper around the original fractal_decode function.
    """
    transformations, img_shape = compressed_data
    # Assuming default range/domain sizes if not provided
    range_size = 8
    domain_size = 16
    return _fractal_decode_internal(transformations, img_shape, range_size, domain_size, iterations)

def fractal_save(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def fractal_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)