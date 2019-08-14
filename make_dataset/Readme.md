# Make Dataset task

This script will load the data, do feature engineering and split the dataset in
train and test

```
Usage: dataset.py [OPTIONS]

  Load the csv, do feature engineering and split in train and test.

  Parameters ---------- name: str     name of the csv file on local disk,
  without '.csv' suffix. in_dir:     directory where csv file should be 
  found. out_dir:     directory where file should be saved to.

  Returns ------- None

Options:
  --name TEXT
  --in-dir TEXT
  --out-dir TEXT
  --help          Show this message and exit.
```
