import numpy as np
import pickle
from tqdm import tqdm

# --- OpenCL GPU Acceleration Dependencies ---
# These libraries are required for cross-vendor GPU support.
# You must install a device driver with OpenCL support and then:
# pip install pyopencl
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

# --- Utility functions (can be reused from CPU version) ---
def get_isometries():
    """Returns a list of 8 functions for block isometries."""
    return [
        lambda block: block, lambda block: np.rot90(block, 1),
        lambda block: np.rot90(block, 2), lambda block: np.rot90(block, 3),
        lambda block: np.fliplr(block), lambda block: np.rot90(np.fliplr(block), 1),
        lambda block: np.rot90(np.fliplr(block), 2), lambda block: np.rot90(np.fliplr(block), 3),
    ]

def downsample_block(block, factor=2):
    """Downsamples a block by an integer factor using local averaging."""
    return block.reshape((block.shape[0]//factor, factor, block.shape[1]//factor, factor)).mean(axis=(1,3))

# --- Fully Implemented OpenCL GPU Kernel ---
# This kernel finds the best matching domain block for a single range block.
OPENCL_KERNEL_SOURCE = """
    #define RANGE_SIZE %d
    #define DOMAIN_SIZE %d
    #define NUM_PIXELS_RANGE (RANGE_SIZE * RANGE_SIZE)
    #define DOWNSAMPLE_FACTOR (DOMAIN_SIZE / RANGE_SIZE)

    // Function to apply one of 8 isometries to a block
    void apply_isometry(float block[RANGE_SIZE][RANGE_SIZE], int iso_idx) {
        float temp[RANGE_SIZE][RANGE_SIZE]; // Use a temporary buffer to avoid read/write conflicts
        for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) temp[i][j] = block[i][j];

        // Each case directly maps the temp buffer to the block buffer with the correct transformation.
        // This is safer and clearer than sequential rotations and flips.
        switch(iso_idx) {
            case 0: break; // Identity
            case 1: // rot90
                for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) block[i][j] = temp[RANGE_SIZE-1-j][i];
                break;
            case 2: // rot180
                for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) block[i][j] = temp[RANGE_SIZE-1-i][RANGE_SIZE-1-j];
                break;
            case 3: // rot270
                for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) block[i][j] = temp[j][RANGE_SIZE-1-i];
                break;
            case 4: // fliplr
                for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) block[i][j] = temp[i][RANGE_SIZE-1-j];
                break;
            case 5: // fliplr -> rot90
                for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) block[i][j] = temp[j][i];
                break;
            case 6: // fliplr -> rot180
                for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) block[i][j] = temp[RANGE_SIZE-1-i][j];
                break;
            case 7: // fliplr -> rot270
                for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) block[i][j] = temp[RANGE_SIZE-1-j][RANGE_SIZE-1-i];
                break;
        }
    }

    __kernel void find_best_match(
        __global const float* img,
        const int height,
        const int width,
        __global const int* range_coords,      // Input: Coordinates of range blocks
        __global const int* domain_coords,     // Input: Coordinates of domain blocks
        const int num_domain_blocks,
        __global float* transformations_out)   // Output: (ri, rj, di, dj, a, b, iso_idx)
    {
        int gid = get_global_id(0);

        // 1. Get Range Block info for this kernel instance
        int ri = range_coords[gid * 2];
        int rj = range_coords[gid * 2 + 1];

        // 2. Load Range Block into private memory for fast access
        float range_block[RANGE_SIZE][RANGE_SIZE];
        float range_flat[NUM_PIXELS_RANGE];
        float sum_y = 0.0f;
        int k = 0;
        for (int i = 0; i < RANGE_SIZE; i++) {
            for (int j = 0; j < RANGE_SIZE; j++) {
                float val = img[(ri + i) * width + (rj + j)];
                range_block[i][j] = val;
                range_flat[k++] = val;
                sum_y += val;
            }
        }
        float mean_y = sum_y / NUM_PIXELS_RANGE;

        // 3. Initialize search variables
        float min_err = FLT_MAX;
        float best_di = -1, best_dj = -1, best_a = 0, best_b = 0;
        int best_iso_idx = -1;

        // 4. Iterate through all domain blocks
        for (int d_idx = 0; d_idx < num_domain_blocks; d_idx++) {
            int di = domain_coords[d_idx * 2];
            int dj = domain_coords[d_idx * 2 + 1];

            // 5. Downsample domain block into private memory
            float domain_ds[RANGE_SIZE][RANGE_SIZE];
            for (int i = 0; i < RANGE_SIZE; i++) {
                for (int j = 0; j < RANGE_SIZE; j++) {
                    float sum = 0.0f;
                    for (int y = 0; y < DOWNSAMPLE_FACTOR; y++) {
                        for (int x = 0; x < DOWNSAMPLE_FACTOR; x++) {
                            sum += img[((di + i * DOWNSAMPLE_FACTOR) + y) * width + (dj + j * DOWNSAMPLE_FACTOR + x)];
                        }
                    }
                    domain_ds[i][j] = sum / (DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR);
                }
            }

            // 6. Iterate through all 8 isometries
            for (int iso_idx = 0; iso_idx < 8; iso_idx++) {
                float candidate[RANGE_SIZE][RANGE_SIZE];
                for(int i=0; i<RANGE_SIZE; i++) for(int j=0; j<RANGE_SIZE; j++) candidate[i][j] = domain_ds[i][j];
                apply_isometry(candidate, iso_idx);

                // 7. Calculate contrast (a) and brightness (b)
                float sum_x = 0.0f, sum_xx = 0.0f, sum_xy = 0.0f;
                k = 0;
                for (int i = 0; i < RANGE_SIZE; i++) {
                    for (int j = 0; j < RANGE_SIZE; j++) {
                        float x_val = candidate[i][j];
                        float y_val = range_block[i][j];
                        sum_x += x_val;
                        sum_xx += x_val * x_val;
                        sum_xy += x_val * y_val;
                    }
                }
                float mean_x = sum_x / NUM_PIXELS_RANGE;
                float var_x = (sum_xx / NUM_PIXELS_RANGE) - (mean_x * mean_x);

                if (var_x < 1e-6f) continue;

                float cov_xy = (sum_xy / NUM_PIXELS_RANGE) - (mean_x * mean_y);
                float a = cov_xy / var_x;
                float b = mean_y - a * mean_x;

                // 8. Calculate MSE
                float err = 0.0f;
                // Correctly iterate over the 2D block to calculate error
                for (int i = 0; i < RANGE_SIZE; i++) {
                    for (int j = 0; j < RANGE_SIZE; j++) {
                        float pred = a * candidate[i][j] + b;
                        float diff = range_block[i][j] - pred;
                        err += diff * diff;
                    }
                }
                err /= NUM_PIXELS_RANGE;

                // 9. Check if this is the best match so far
                if (err < min_err) {
                    min_err = err;
                    best_di = di;
                    best_dj = dj;
                    best_a = a;
                    best_b = b;
                    best_iso_idx = iso_idx;
                }
            }
        }

        // 10. Write best parameters to global output buffer
        int out_idx = gid * 7;
        transformations_out[out_idx + 0] = ri;
        transformations_out[out_idx + 1] = rj;
        transformations_out[out_idx + 2] = best_di;
        transformations_out[out_idx + 3] = best_dj;
        transformations_out[out_idx + 4] = best_a;
        transformations_out[out_idx + 5] = best_b;
        transformations_out[out_idx + 6] = best_iso_idx;
    }
"""

def fractal_compress_opencl(img_color, range_size=8, domain_size=16, show_progress=True):
    """
    Compresses a color image using a fractal algorithm accelerated by OpenCL.
    This function acts as the "host" code that sets up data and launches the GPU "kernel".
    """
    if not OPENCL_AVAILABLE:
        raise ImportError("pyopencl is not installed. OpenCL acceleration is not available.")

    # --- Host Code ---
    # 1. Convert image to grayscale and float32, same as the CPU version.
    img = np.dot(img_color[...,:3], [0.2989, 0.5870, 0.1140])
    height, width = img.shape
    img = img.astype(np.float32) / 255.0

    # 2. Prepare coordinates for range and domain blocks on the CPU
    range_coords_list = [(i, j) for i in range(0, height - range_size + 1, range_size) for j in range(0, width - range_size + 1, range_size)]
    num_range_blocks = len(range_coords_list)
    range_coords_np = np.array(range_coords_list, dtype=np.int32)

    domain_stride = range_size // 2
    domain_coords_list = [(i, j) for i in range(0, height - domain_size + 1, domain_stride) for j in range(0, width - domain_size + 1, domain_stride)]
    num_domain_blocks = len(domain_coords_list)
    domain_coords_np = np.array(domain_coords_list, dtype=np.int32)

    # 3. Set up OpenCL context, command queue, and compile the kernel.
    try:
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)
        print(f"[INFO] Selected device: {queue.device.name}")
    except cl.LogicError:
        raise RuntimeError("No OpenCL-capable GPU found. Please check your drivers.")

    # 4. Format the kernel source with constants and compile it.
    formatted_kernel = OPENCL_KERNEL_SOURCE % (range_size, domain_size)
    prg = cl.Program(ctx, formatted_kernel).build()

    # 5. Create GPU memory buffers and transfer data from CPU to GPU.
    mf = cl.mem_flags
    img_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    range_coords_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=range_coords_np)
    domain_coords_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=domain_coords_np)
    transformations_out_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_range_blocks * 7 * np.dtype(np.float32).itemsize)

    # 6. Launch the Kernel.
    # The global work size is the number of range blocks. Each kernel instance processes one.
    pbar = tqdm(total=num_range_blocks, desc="Fractal Encoding (OpenCL)", unit="block", disable=not show_progress)
    
    # Get a handle to the kernel function
    kernel = prg.find_best_match

    # Set kernel arguments
    kernel.set_args(
        img_gpu,
        np.int32(height),
        np.int32(width),
        range_coords_gpu,
        domain_coords_gpu,
        np.int32(num_domain_blocks),
        transformations_out_gpu
    )
    
    # Execute the kernel
    cl.enqueue_nd_range_kernel(queue, kernel, (num_range_blocks,), None)
    
    # 7. Transfer results back from GPU to CPU.
    # Create an empty numpy array to receive the data.
    transformations_np = np.empty((num_range_blocks, 7), dtype=np.float32)
    cl.enqueue_copy(queue, transformations_np, transformations_out_gpu).wait()
    pbar.update(num_range_blocks)
    pbar.close()

    # 8. Convert the NumPy array of transformations into the list of tuples format.
    # The kernel output includes ri and rj, which we can discard as they are implicit.
    transformations = [
        (int(t[0]), int(t[1]), int(t[2]), int(t[3]), t[4], t[5], int(t[6]))
        for t in transformations_np
    ]

    print("[INFO] OpenCL processing finished.")
    return transformations

# Decompression is typically fast and often left on the CPU.
# We can reuse the structured CPU decompressor.
def fractal_decompress_opencl(transformations, img_shape, range_size=8, domain_size=16, iterations=10):
    """
    Decompresses an image from a set of fractal transformations on the CPU.
    This is generally fast enough and avoids GPU-CPU data transfer overhead for each iteration.
    """
    height, width = img_shape[:2]
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

        counts[counts == 0] = 1
        img = new_img / counts

    return np.clip(img * 255, 0, 255).astype(np.uint8)
