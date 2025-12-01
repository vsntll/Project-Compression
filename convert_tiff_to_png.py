import os
import imageio.v2 as iio
from tqdm import tqdm

def convert_tiff_in_database(root_dirs=["data"]):
    """
    Walks through all subdirectories of the given root directories, finds all .tif and .tiff
    files, and converts them to .png format in the same folder.
    """
    tiff_files = []
    # First, collect all TIFF files to have a total for the progress bar
    for root_dir in root_dirs:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    tiff_files.append(os.path.join(subdir, file))

    if not tiff_files:
        print(f"No .tif or .tiff files were found in the specified directories: {root_dirs}")
        return

    print(f"Found {len(tiff_files)} TIFF image(s) to convert to PNG.")

    # Process the collected files with a progress bar
    for tiff_path in tqdm(tiff_files, desc="Converting TIFF to PNG", unit="file"):
        try:
            # Define the output path for the new PNG file
            png_path = os.path.splitext(tiff_path)[0] + '.png'

            # Read the TIFF image data
            image_data = iio.imread(tiff_path)

            # Write the data to a new PNG file
            iio.imwrite(png_path, image_data)

            # Uncomment the line below if you want to delete the original .tiff file after conversion
            # os.remove(tiff_path)

        except Exception as e:
            # Print an error message if a specific file fails to convert
            print(f"\n[ERROR] Failed to convert {tiff_path}: {e}")

    print("\nConversion process finished.")

if __name__ == "__main__":
    # The main data directories for your project, now including results
    database_roots = ["data", "results"]
    convert_tiff_in_database(database_roots)