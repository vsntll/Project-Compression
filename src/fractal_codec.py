import numpy as np
import cv2
import pickle

def downsample_block(block, factor=2):
    return block.reshape((block.shape[0]//factor, factor, block.shape[1]//factor, factor)).mean(axis=(1,3))

def fractal_encode(img, range_size=8, domain_size=16):
    height, width = img.shape
    img = img.astype(np.float32) / 255.0
    transformations = []  # list of (range_x, range_y, domain_x, domain_y, contrast, brightness, flip_angle)

    for i in range(0, height - range_size + 1, range_size):
        for j in range(0, width - range_size + 1, range_size):
            range_block = img[i:i+range_size, j:j+range_size]
            min_err = float('inf')
            best_params = None

            for di in range(0, height - domain_size + 1, range_size//2):
                for dj in range(0, width - domain_size + 1, range_size//2):
                    domain_block = img[di:di+domain_size, dj:dj+domain_size]
                    domain_ds = downsample_block(domain_block, factor=domain_size//range_size)

                    # Transform candidates: no flip/rot only for simplicity here; extend for full method
                    candidate = domain_ds

                    x = candidate.flatten()
                    y = range_block.flatten()
                    var_x = np.var(x)
                    if var_x == 0:
                        continue
                    a = np.cov(x,y)[0,1] / var_x  # contrast
                    b = np.mean(y) - a*np.mean(x) # brightness

                    pred = a*x + b
                    err = np.mean((y - pred)**2)

                    if err < min_err:
                        min_err = err
                        best_params = (i, j, di, dj, a, b)

            transformations.append(best_params)
    return transformations

def fractal_save(transformations, filename):
    with open(filename, 'wb') as f:
        pickle.dump(transformations, f)

def fractal_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def fractal_decode(transformations, img_shape, range_size=8, domain_size=16, iterations=10):
    height, width = img_shape
    img = np.zeros(img_shape, dtype=np.float32)

    for _ in range(iterations):
        new_img = np.zeros_like(img)
        counts = np.zeros_like(img)

        for (ri, rj, di, dj, a, b) in transformations:
            # get domain block and downsample
            domain_block = img[di:di+domain_size, dj:dj+domain_size]
            domain_ds = downsample_block(domain_block, factor=domain_size//range_size)

            block_pred = a * domain_ds + b
            block_pred = np.clip(block_pred, 0, 1)

            new_img[ri:ri+range_size, rj:rj+range_size] += block_pred
            counts[ri:ri+range_size, rj:rj+range_size] += 1

        counts[counts==0] = 1
        img = new_img / counts

    img_uint8 = (img * 255).astype(np.uint8)
    return img_uint8

# Usage example
if __name__ == "__main__":
    img = cv2.imread('path/to/grayscale_image.png', cv2.IMREAD_GRAYSCALE)
    trans = fractal_encode(img)
    fractal_save(trans, 'fractal_params.pkl')
    loaded_trans = fractal_load('fractal_params.pkl')
    recon_img = fractal_decode(loaded_trans, img.shape, iterations=15)
    cv2.imwrite('reconstructed_fractal.png', recon_img)
