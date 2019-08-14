import click
from dask.distributed import Client
import dask.dataframe as dd
from pathlib import Path
import joblib
from joblib import dump
from sklearn.ensemble import RandomForestClassifier


@click.command()
@click.option('--in-parquet')
@click.option('--out-dir')
@click.option('--in-dir')
def train_model(in_parquet, out_dir, in_dir):
    """Load a parquet file and train the model.

    Parameters
    ----------
    in-parquet: str
        name of the parquet file on local disk, without '.parquet' suffix.
    in-dir:
        directory where parquet file should be found.
    out_dir:
        directory where files should be saved to.

    Returns
    -------
    None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)

    clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                                 criterion='gini', max_depth=None,
                                 max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None,
                                 min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0,
                                 n_estimators=100, n_jobs=None,
                                 oob_score=False, random_state=None,
                                 verbose=0, warm_start=False)

    # load train set as dask Dataframe
    ddf = dd.read_parquet(f'{in_dir}/{in_parquet}.parquet',
                          engine='fastparquet')

    # create features and labels
    y = ddf['wine_rating']
    X = ddf.drop(['wine_rating'], axis=1)

    clf.fit(X, y)

    # serialize the model
    dump(clf, f'{out_dir}/trained_model.joblib')

    # flag to point the process has been concluded
    flag = out_dir / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    train_model()