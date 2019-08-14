# Train Model task

This script will load the parquet file, train the model and create a serialized
model.

```
Usage: train.py [OPTIONS]

  Load a parquet file and train the model.

  Parameters ---------- in_parquet: str     name of the parquet file on local
  disk, without '.parquet' suffix. in_dir:     directory where parquet file 
  should be found. out_dir:     directory where file should be saved to.

  Returns ------- None

Options:
  --in-parquet TEXT
  --in-dir TEXT
  --out-dir TEXT
  --help          Show this message and exit.
```
