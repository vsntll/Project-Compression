import numpy as np
import pickle
from tqdm import tqdm

def get_isometries():
    """Returns a list of 8 functions for block isometries."""
    return [
        lambda block: block,
        lambda block: np.rot90(block, 1),
        lambda block: np.rot90(block, 2),
        lambda block: np.rot90(block, 3),
        lambda block: np.fliplr(block),
        lambda block: np.rot90(np.fliplr(block), 1),
        lambda block: np.rot90(np.fliplr(block), 2),
        lambda block: np.rot90(np.fliplr(block), 3),
    ]

def downsample_block(block, factor=2):
    """Downsamples a block by an integer factor using local averaging."""
    return block.reshape((block.shape[0]//factor, factor, block.shape[1]//factor, factor)).mean(axis=(1,3))

def classify_block(block, num_classes=3):
    """Classifies a block based on its variance to speed up search."""
    variance = np.var(block)
    if variance < 0.01: return 0  # Flat
    elif variance < 0.05: return 1  # Medium variance
    else: return 2  # High variance

def fractal_compress(img_color, range_size=8, domain_size=16, show_progress=True, position=0, distortion_threshold=0.01):
    """
    Compresses a color image using a structured fractal algorithm.
    This version is optimized with block classification and isometries.
    """
    # Convert to grayscale for encoding
    img = np.dot(img_color[...,:3], [0.2989, 0.5870, 0.1140])
    height, width = img.shape
    img = img.astype(np.float32) / 255.0
    
    transformations = []
    isometries = get_isometries()

    # Pre-classify all possible domain blocks
    domain_blocks_by_class = {i: [] for i in range(3)}
    step = range_size // 2  # Overlapping domain blocks
    for di in range(0, height - domain_size + 1, step):
        for dj in range(0, width - domain_size + 1, step):
            domain_block = img[di:di+domain_size, dj:dj+domain_size]
            block_class = classify_block(domain_block)
            domain_blocks_by_class[block_class].append((di, dj, domain_block))

    range_blocks_coords = [(i, j) for i in range(0, height, range_size) for j in range(0, width, range_size)]
    
    pbar_desc = "Fractal Encoding (Structured)"
    pbar = tqdm(range_blocks_coords, desc=pbar_desc, unit="block", disable=not show_progress, position=position, leave=False)
    
    for i, j in pbar:
        range_block = img[i:i+range_size, j:j+range_size]
        min_err = float('inf')
        best_params = None

        range_block_class = classify_block(range_block)

        # Search within the corresponding class of domain blocks
        for di, dj, domain_block in domain_blocks_by_class[range_block_class]:
            domain_ds = downsample_block(domain_block, factor=domain_size//range_size)

            for iso_idx, isometry in enumerate(isometries):
                candidate = isometry(domain_ds)
                
                x = candidate.flatten()
                y = range_block.flatten()
                var_x = np.var(x)
                if var_x < 1e-6: continue

                a = np.cov(x, y)[0, 1] / var_x
                b = np.mean(y) - a * np.mean(x)

                pred = a * candidate + b
                err = np.mean((range_block - pred)**2)

                if err < min_err:
                    min_err = err
                    best_params = (i, j, di, dj, a, b, iso_idx)
            
            # Early exit if a good enough match is found
            if min_err < distortion_threshold:
                break
        
        if best_params:
            transformations.append(best_params)
            
    return transformations

def fractal_decompress(transformations, img_shape, range_size=8, domain_size=16, iterations=10):
    """Decompresses an image from a set of fractal transformations."""
    height, width = img_shape[:2] # Handle color or grayscale shape
    img = np.zeros((height, width), dtype=np.float32)
    isometries = get_isometries()

    for _ in range(iterations):
        new_img = np.zeros_like(img)
        counts = np.zeros_like(img, dtype=int)

        for (ri, rj, di, dj, a, b, iso_idx) in transformations:
            domain_block = img[di:di+domain_size, dj:dj+domain_size]
            domain_ds = downsample_block(domain_block, factor=domain_size//range_size)
            
            transformed_domain = isometries[iso_idx](domain_ds)
            
            block_pred = a * transformed_domain + b
            
            new_img[ri:ri+range_size, rj:rj+range_size] += block_pred
            counts[ri:ri+range_size, rj:rj+range_size] += 1

        # Average overlapping blocks
        counts[counts == 0] = 1
        img = new_img / counts

    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def fractal_save(transformations, filename):
    """Saves transformation data using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(transformations, f)

def fractal_load(filename):
    """Loads transformation data using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)