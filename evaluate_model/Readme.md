# Evaluate Model task

This script will load a parquet file and serialized model, evaluate the results.

```
Usage: train.py [OPTIONS]

  Load a parquet file and train the model.

  Parameters ---------- in_parquet: str     name of the parquet file on local
  disk, without '.parquet' suffix. in_dir:     directory where parquet file 
  should be found. out_dir:     directory where file should be saved to. 
  model_name:     name of the joblib file, without the '.joblib' suffix.


  Returns ------- None

Options:
  --in-parquet TEXT
  --in-dir TEXT
  --out-dir TEXT
  --model-name TEXT
  --help          Show this message and exit.
```
