import click
from pathlib import Path
import pweave
import pandas as pd
import pandas_profiling


@click.command()
@click.option('--in-parquet')
@click.option('--in-csv')
@click.option('--out-dir')
@click.option('--in-dir')
@click.option('--model-name')
def evaluate_model(in_parquet, in_csv, out_dir, in_dir, model_name):
    """Load a parquet file and serialized model, evaluate the results.

    Parameters
    ----------
    in-parquet: str
        name of the parquet file on local disk, without '.parquet' suffix.
    in-dir:
        directory where parquet file should be found.
    out_dir:
        directory where files should be saved to.
    model_name:
        name of the joblib file, without the '.joblib' suffix.

    Returns
    -------
    None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)

    # create dataset profile
    df = pd.read_csv(f'{in_dir}/raw/{in_csv}.csv')
    profile = df.profile_report(title='Wine Rating Prediction Dataset',
                                style={'full_width': True})
    profile.to_file(output_file=f'{out_dir}/dataset_profile.html')

    # create report
    pweave.publish('report.pmd', doc_format='html', theme='skeleton',
                   latex_engine='pdflatex', output=f'{out_dir}/report.html')

    # flag to point the process has been concluded
    flag = out_dir / '.SUCCESS'
    flag.touch()

if __name__ == '__main__':
    evaluate_model()
