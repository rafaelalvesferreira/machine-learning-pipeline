import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train.parquet/'
    out_test = outdir / 'test.parquet/'
    flag = outdir / '.SUCCESS'

    train.to_parquet(str(out_train))
    test.to_parquet(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
@click.option('--in-dir')
def make_datasets(in_csv, out_dir, in_dir):
    """Load a csv file, do feature engineering and split in train and test.

    Parameters
    ----------
    name: str
        name of the csv file on local disk, without '.csv' suffix.
    in-dir:
        directory where csv file should be found.
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

    # Connect to the dask cluster
    # c = Client('dask-scheduler:8786')

    # load data as a dask Dataframe if you have trouble with dask
    # please fall back to pandas or numpy
    ddf = dd.read_csv(f'{in_dir}/{in_csv}', blocksize=1e6)

    # we set the index so we can properly execute loc below
    ddf = ddf.set_index('Unnamed: 0')

    # TODO: implement proper dataset creation here
    # http://docs.dask.org/en/latest/dataframe-api.html
    # split dataset into train test feel free to adjust test percentage

    # drop of categorical features that have a high percentage of NaN
    ddf = ddf.drop(['description', 'designation', 'region_1', 'region_2',
                    'taster_twitter_handle', 'winery'], axis=1)

    # calculate the quantiles of points to be the rating standards
    min_point = ddf['points'].min().compute()
    rating_one = ddf['points'].quantile(0.1).compute()
    rating_two = ddf['points'].quantile(0.3).compute()
    rating_three = ddf['points'].quantile(0.7).compute()
    rating_four = ddf['points'].quantile(0.9).compute()
    max_point = ddf['points'].max().compute()

    # function to apply the rating to the points feature
    def rating_the_points(points):
        if points <= rating_one:
            return 1
        if points <= rating_two:
            return 2
        if points <= rating_three:
            return 3
        if points <= rating_four:
            return 4
        if points <= max_point:
            return 5
        else:
            raise ValueError('Input value out of range')

    # apply function rating_the_points to de dataframe
    ddf['wine_rating'] = ddf.loc[:, 'points'].apply(
        lambda x: rating_the_points(x), meta=('points', 'int64')).compute()

    # change types to categorical
    # transform categorical data in new feaures
    ddf = ddf.categorize()
    ddf = dd.get_dummies(ddf)

    # trigger computation
    n_samples = len(ddf)

    idx = np.arange(n_samples)
    test_idx = idx[:n_samples // 10]
    test = ddf.loc[test_idx]
    # drop nan values from test set
    test = test.dropna()

    train_idx = idx[n_samples // 10:]
    train = ddf.loc[train_idx]
    # drop nan values from train set
    train = train.dropna()

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
