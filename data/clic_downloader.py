import tensorflow_datasets as tfds

# Download and prepare CLIC dataset
clic_dataset = tfds.load('clic', split='train', shuffle_files=False)

# To download the whole dataset, use 'all' or other splits such as 'test' or 'validation'
clic_all = tfds.load('clic', split='all', shuffle_files=False)
