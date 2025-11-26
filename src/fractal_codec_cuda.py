import cupy as cp
import pickle
from tqdm import tqdm
import cv2

# --- Core Fractal Functions (GPU-accelerated with CuPy) ---

def _downsample_block_gpu(block, factor=2):
    """Downsamples a block by an integer factor on the GPU."""
    # Reshape and then calculate the mean. CuPy handles this efficiently on the GPU.
    return block.reshape((block.shape[0] // factor, factor, block.shape[1] // factor, factor)).mean(axis=(1, 3))

def _fractal_encode_gpu_internal(img_gpu, range_size, domain_size, show_progress, distortion_threshold, position=0):
    """Internal fractal encoding logic optimized for GPU with CuPy."""
    height, width = img_gpu.shape
    transformations = []

    # --- Prepare Domain Blocks on GPU ---
    domain_stride = range_size // 2
    domain_grid_h = (height - domain_size) // domain_stride + 1
    domain_grid_w = (width - domain_size) // domain_stride + 1
    
    # Create a batch of all possible downsampled domain blocks on the GPU
    num_domain_blocks = domain_grid_h * domain_grid_w
    all_domain_ds = cp.empty((num_domain_blocks, range_size, range_size), dtype=cp.float32)
    domain_coords = cp.empty((num_domain_blocks, 2), dtype=cp.int32)

    idx = 0
    for i in range(domain_grid_h):
        for j in range(domain_grid_w):
            di, dj = i * domain_stride, j * domain_stride
            domain_block = img_gpu[di:di+domain_size, dj:dj+domain_size]
            all_domain_ds[idx] = _downsample_block_gpu(domain_block, factor=domain_size // range_size)
            domain_coords[idx] = cp.array([di, dj])
            idx += 1

    # Pre-calculate stats for all domain blocks in a vectorized way
    domain_flat = all_domain_ds.reshape(num_domain_blocks, -1)
    mean_x = domain_flat.mean(axis=1)
    var_x = domain_flat.var(axis=1)

    range_blocks_coords = [(i, j) for i in range(0, height - range_size + 1, range_size) for j in range(0, width - range_size + 1, range_size)]
    
    pbar = tqdm(range_blocks_coords, desc=f"GPU Worker {position}", unit="block", disable=not show_progress, leave=False, position=position)
    
    for ri, rj in pbar:
        range_block = img_gpu[ri:ri+range_size, rj:rj+range_size]
        y_flat = range_block.flatten()
        mean_y = y_flat.mean()

        # --- Vectorized Block Matching on GPU ---
        # Calculate covariance for all domain blocks against the current range block at once
        # cov(x,y) = E[xy] - E[x]E[y]
        cov_xy = cp.mean(domain_flat * y_flat, axis=1) - (mean_x * mean_y)

        # Calculate contrast (a) and brightness (b) for all domain blocks
        # Add a small epsilon to var_x to avoid division by zero
        a = cov_xy / (var_x + 1e-7)
        b = mean_y - a * mean_x

        # Calculate error for all blocks simultaneously
        # Error(y, ax+b) = E[y^2] - 2aE[xy] - 2bE[y] + a^2E[x^2] + 2abE[x] + b^2
        # A simpler way is to compute the prediction and then the MSE, but it uses more memory.
        # Let's use the direct MSE calculation for memory efficiency.
        # pred = a[:, None, None] * all_domain_ds + b[:, None, None]
        # err = cp.mean((range_block - pred)**2, axis=(1, 2))
        
        # This is a more direct error calculation that can be faster
        # err = var(y) + (a^2*var(x) - 2*a*cov(x,y))
        err = y_flat.var() + (a**2 * var_x) - (2 * a * cov_xy)

        # Find the best domain block (minimum error)
        best_idx = cp.argmin(err)
        min_err = err[best_idx]
        
        best_di, best_dj = domain_coords[best_idx]
        best_a = a[best_idx]
        best_b = b[best_idx]

        # Transfer results from GPU to CPU for storage
        transformations.append((ri, rj, int(best_di), int(best_dj), float(best_a), float(best_b)))

        # Early exit if a good enough match is found
        if min_err < distortion_threshold:
            continue
            
    return transformations

# --- Public API Functions ---

def fractal_compress_cuda(img, range_size=8, domain_size=16, show_progress=True, distortion_threshold=0.001, position=0):
    """
    Compresses a grayscale image using fractal encoding, accelerated on the GPU with CuPy.
    """
    if len(img.shape) > 2 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Move image to GPU and convert to float
    img_gpu = cp.asarray(img_gray, dtype=cp.float32) / 255.0

    transformations = _fractal_encode_gpu_internal(img_gpu, range_size, domain_size, show_progress, distortion_threshold, position)
    
    return transformations, img_gray.shape

def _fractal_decode_gpu_internal(transformations, img_shape, range_size, domain_size, iterations):
    """Internal fractal decoding logic accelerated on the GPU with CuPy."""
    height, width = img_shape
    # Start with a black image on the GPU
    img_gpu = cp.zeros(img_shape, dtype=cp.float32)

    for _ in range(iterations):
        new_img_gpu = cp.zeros_like(img_gpu)
        for (ri, rj, di, dj, a, b) in transformations:
            domain_block = img_gpu[di:di+domain_size, dj:dj+domain_size]
            domain_ds = _downsample_block_gpu(domain_block, factor=domain_size // range_size)
            
            # Apply transformation and clip on GPU
            block_pred = cp.clip(a * domain_ds + b, 0, 1)
            new_img_gpu[ri:ri+range_size, rj:rj+range_size] = block_pred
        img_gpu = new_img_gpu

    # Convert final image back to CPU memory as a uint8 array
    return (img_gpu * 255).get().astype('uint8')

def fractal_decompress_cuda(compressed_data, iterations=10):
    transformations, img_shape = compressed_data
    range_size = 8
    domain_size = 16
    return _fractal_decode_gpu_internal(transformations, img_shape, range_size, domain_size, iterations)

def fractal_save(data, filename):
    """Saves compressed data (transformations, shape) to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def fractal_load(filename):
    """Loads compressed data from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)