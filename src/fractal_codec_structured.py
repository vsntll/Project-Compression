import numpy as np
import cv2
import pickle
from tqdm import tqdm

# --- Core Fractal Functions (self-contained) ---

def _downsample_block(block, factor=2):
    """Downsamples a block by an integer factor."""
    return block.reshape((block.shape[0]//factor, factor, block.shape[1]//factor, factor)).mean(axis=(1,3))

def _classify_block(block, num_classes=3):
    """Classifies a block based on its variance."""
    variance = np.var(block)
    if variance < 0.01:
        return 0  # Flat class
    elif variance < 0.05:
        return 1  # Medium variance class
    else:
        return 2  # High variance class

def _fractal_encode_internal(img, range_size, domain_size, show_progress):
    """Internal fractal encoding logic."""
    height, width = img.shape
    img_float = img.astype(np.float32) / 255.0
    transformations = []

    domain_blocks_by_class = {i: [] for i in range(3)}
    for di in range(0, height - domain_size + 1, range_size // 2):
        for dj in range(0, width - domain_size + 1, range_size // 2):
            domain_block = img_float[di:di+domain_size, dj:dj+domain_size]
            block_class = _classify_block(domain_block)
            domain_blocks_by_class[block_class].append((di, dj, domain_block))

    range_blocks_coords = [(i, j) for i in range(0, height - range_size + 1, range_size) for j in range(0, width - range_size + 1, range_size)]
    
    pbar = tqdm(range_blocks_coords, desc="Fractal Encoding", unit="block", disable=not show_progress, leave=False)
    for i, j in pbar:
        range_block = img_float[i:i+range_size, j:j+range_size]
        min_err = float('inf')
        best_params_for_block = None
        range_block_class = _classify_block(range_block)

        for di, dj, domain_block in domain_blocks_by_class[range_block_class]:
            domain_ds = _downsample_block(domain_block, factor=domain_size // range_size)
            candidate = domain_ds
            x = candidate.flatten()
            y = range_block.flatten()
            var_x = np.var(x)
            if var_x < 1e-6:
                continue
            a = np.cov(x, y)[0, 1] / var_x
            b = np.mean(y) - a * np.mean(x)
            pred = a * candidate + b
            err = np.mean((range_block - pred)**2)
            if err < min_err:
                min_err = err
                best_params_for_block = (i, j, di, dj, a, b)

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
def fractal_compress(img, range_size=8, domain_size=16, show_progress=False):
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
    transformations = _fractal_encode_internal(img_gray, range_size, domain_size, show_progress)
    
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