# Project Compression

A high-performance analysis pipeline for comparing classical image compression techniques: **Wavelet Transforms** vs. **CUDA-Accelerated Fractal Encoding**.

This project benchmarks compression performance across standard image datasets, with a highly optimized fractal encoding implementation that leverages both CPU and GPU workers in parallel. It outputs reconstructed images and generates comprehensive metrics (PSNR, SSIM, bitrate) for quantitative comparison.

## Overview

Image compression is fundamental to digital media, balancing quality and file size. This project implements and compares two classical compression approaches:

- **Wavelet-based compression**: Multi-resolution decomposition with quantization
- **Fractal compression**: Self-similarity-based encoding using iterative block matching, now heavily accelerated with a hybrid CPU/GPU parallel pipeline.

The pipeline processes images from multiple datasets, applies both compression methods, reconstructs the images, and exports detailed performance metrics for analysis.

## Features

- Batch processing of multiple image datasets
- Complete compression/decompression workflows for both methods
- **CUDA-Accelerated Fractal Codec**: Core fractal compression and decompression logic is accelerated with CUDA via CuPy for massive performance gains.
- **Hybrid CPU/GPU Parallel Processing**: For fractal encoding, the system leverages all available CPU cores and multiple GPU workers simultaneously to maximize throughput.
- **Dynamic Task Scheduling & Work-Stealing**: A global task queue feeds all workers. Idle GPU workers can "steal" and accelerate tasks from busy CPU workers, ensuring high resource utilization and minimizing total runtime.
- **GPU Task Prioritization**: Larger, more computationally intensive images are automatically sorted and assigned to GPU workers first.
- Automated quality metrics calculation (PSNR, SSIM, bitrate)
- Visual outputs reconstructed images saved for inspection
- CSV export of all metrics for further analysis
- Real-time progress tracking with `tqdm` progress bars
- Modular architecture for easy extension.

## Project Structure

```
Project-Compression/
├── data/                       # Input image datasets
│   ├── clic_subset/           # CLIC dataset subset
│   ├── kodak/                 # Kodak PhotoCD test images
│   └── standard_test/         # Standard test images (Lena, Peppers, etc.)
├── results/                   # Output directory
│   ├── fractal_cuda_parallel/ # Fractal-compressed results from the hybrid pipeline
│   │   ├── clic_subset/
│   │   ├── kodak/
│   │   └── standard_test/
│   └── metrics/               # Performance metrics CSV
├── src/                       # Compression codec modules
│   ├── fractal_codec_cuda.py      # CUDA-accelerated fractal implementation
│   ├── fractal_codec_structured.py# CPU-based fractal implementation
│   │   ├── clic_subset/
│   │   ├── kodak/
│   │   └── standard_test/
│   └── metrics/              # Performance metrics CSV
├── src/                      # Compression codec modules
│   ├── fractal_codec.py     # Fractal compression implementation
│   └── wavelet_codec.py     # Wavelet compression implementation
├── fractal_structured_test.py # Main script for the advanced fractal pipeline
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- An NVIDIA GPU with CUDA Toolkit installed (version 11.x or 12.x recommended)
- pip package manager

### Dependencies

**Note on CuPy:** The CUDA-accelerated fractal codec depends on CuPy, which must be installed according to your specific CUDA Toolkit version. It is highly recommended to install it manually *before* installing the other requirements.

For example, for CUDA 12.x:
```bash
pip install cupy-cuda12x
```
See the [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html) for other versions.

```bash
pip install -r requirements.txt
```

## Datasets

The project supports multiple standard image compression benchmarking datasets:

### Kodak PhotoCD Dataset
- 24 high-quality RGB images (768×512 pixels)
- Industry-standard benchmark for lossy compression
- Download: [Kodak Lossless True Color Image Suite](http://r0k.us/graphics/kodak/)

### USC-SIPI Standard Test Images
- Classic test images: Lena, Peppers, Baboon, Barbara, etc.
- Available in 256×256 and 512×512 resolutions
- Download: [USC-SIPI Misc Volume](https://sipi.usc.edu/database/database.php?volume=misc)

### CLIC (Challenge on Learned Image Compression)
- Professional and mobile photography datasets
- Use `clic_downloader.py` and `clic_saver.py` to obtain a subset
- Source: [CLIC via TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/clic)

Place downloaded images in the corresponding folders under `data/`.

## Usage

### Basic Usage

Run the main processing pipeline:

```bash
python main.py
```

This will:
1. Process all images in the configured datasets
2. Apply wavelet and fractal compression
3. Save reconstructed images
4. Export metrics to CSV

### Custom Dataset Configuration

Modify the `data_dirs` dictionary in `main.py`:

```python
data_dirs = {
    "kodak": "data/kodak",
    "standard_test": "data/standard_test",
    "clic_subset": "data/clic_subset"
}
```

### Downloading CLIC Dataset

```bash
# Download CLIC dataset from TensorFlow Datasets
python clic_downloader.py

# Extract images to data/clic_subset/
python clic_saver.py
```

## 📈 Output Files

### Reconstructed Images

Decompressed images are saved for visual quality inspection:

- **Wavelet**: `results/wavelet/<dataset>/<image_name>_wavelet.png`
- **Fractal**: `results/fractal/<dataset>/<image_name>_fractal.png`

### Compressed Representations

- **Wavelet**: `.npz` files containing quantized wavelet coefficients
- **Fractal**: `.pkl` files containing transformation parameters

### Metrics CSV

`results/metrics/compression_comparison_metrics.csv` contains:

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name (kodak, standard_test, clic_subset) |
| `image` | Image filename |
| `psnr_wavelet` | Peak Signal-to-Noise Ratio for wavelet method (dB) |
| `ssim_wavelet` | Structural Similarity Index for wavelet method (0-1) |
| `bitrate_wavelet` | Bits per pixel for wavelet compression |
| `psnr_fractal` | Peak Signal-to-Noise Ratio for fractal method (dB) |
| `ssim_fractal` | Structural Similarity Index for fractal method (0-1) |
| `bitrate_fractal` | Bits per pixel for fractal compression |

## Metrics Explained

### PSNR (Peak Signal-to-Noise Ratio)
- Measures pixel-level difference between original and reconstructed images
- Higher values indicate better quality (typically 30-50 dB for good quality)
- Formula: PSNR = 10 × log₁₀(MAX²/MSE)

### SSIM (Structural Similarity Index)
- Perceptual quality metric considering luminance, contrast, and structure
- Range: -1 to 1 (1 = identical images)
- Better correlates with human perception than PSNR

### Bitrate
- Compression efficiency measured in bits per pixel
- Lower values indicate better compression
- Calculated from compressed file size divided by pixel count

## Customization

### Wavelet Parameters

Edit `src/wavelet_codec.py`:

```python
# Compression parameters
wavelet='haar'       # Wavelet type: 'haar', 'db4', 'sym8', etc.
level=3              # Decomposition levels (higher = finer detail)
quant_step=10        # Quantization step (higher = more compression)
```

### Fractal Parameters

Edit `src/fractal_codec.py`:

```python
# Block matching parameters
range_size=8         # Range block size
domain_size=16       # Domain block size (must be ≥ range_size)
iterations=10        # Decoder iterations
```

## Performance Notes

Processing time varies based on:
- Image count and resolution
- Hardware (CPU, RAM)
- Algorithm complexity

**Typical processing times per image:**
- Wavelet: 0.5-2 seconds
- Fractal: 5-60 seconds (slower due to iterative block matching)

**For full Kodak dataset (24 images):**
- Total time: 3-25 minutes depending on implementation and hardware

**Tip**: Test with 3-5 images first to verify setup before processing full datasets.

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify `src/` folder contains both codec files
- Check that codec files define all required functions at module level

### Long Path Errors (Windows)
- Enable Windows Long Path support: [Instructions](https://pip.pypa.io/warnings/enable-long-paths)
- Or use shorter directory paths (e.g., `C:\clic\`)

### PNG ICC Profile Warnings
- Safe to ignore; TensorFlow warns about non-standard color profiles
- Images are processed correctly despite the warning

## References

### Datasets
- [Kodak Lossless True Color Image Suite](http://r0k.us/graphics/kodak/)
- [USC-SIPI Image Database](https://sipi.usc.edu/database/)
- [CLIC: Challenge on Learned Image Compression](https://www.compression.cc/)

### Methods
- Wavelet Transform Image Compression
- Fractal Image Compression (Barnsley, Jacquin)

## Contributing

Contributions are welcome! Areas for improvement:
- Additional compression methods (JPEG, JPEG2000, etc.)
- Color image support for fractal compression
- GPU acceleration for faster processing
- Additional quality metrics (VIF, MS-SSIM, etc.)
- Visualization tools for results


## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/vsntll/Project-Compression).

---

**Project Status**: Active Development

*Last Updated: November 2025*
