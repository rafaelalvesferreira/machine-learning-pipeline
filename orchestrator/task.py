import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return 'code-challenge/download-data:0.1'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/dataset/')
    in_dir = luigi.Parameter(default='/usr/share/data/raw/')

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return ['python', 'dataset.py',
                '--in-csv', f'{self.fname}.csv',
                '--out-dir', f'{self.out_dir}',
                '--in-dir', f'{self.in_dir}']

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class TrainModel(DockerTask):

    fname = luigi.Parameter(default='train')
    out_dir = luigi.Parameter(default='/usr/share/data/model/')
    in_dir = luigi.Parameter(default='/usr/share/data/dataset/')

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        return ['python', 'train.py',
                '--in-parquet', f'{self.fname}',
                '--out-dir', f'{self.out_dir}',
                '--in-dir', f'{self.in_dir}']

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class EvaluateModel(DockerTask):

    fname = luigi.Parameter(default='test')
    in_csv = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/evaluate/')
    in_dir = luigi.Parameter(default='/usr/share/data/')
    model_name = luigi.Parameter(default='trained_model')

    @property
    def image(self):
        return f'code-challenge/evaluate-model:{VERSION}'

    def requires(self):
        return TrainModel()

    @property
    def command(self):
        return ['python', 'evaluate.py',
                '--in-parquet', f'{self.fname}',
                '--in-csv', f'{self.in_csv}',
                '--out-dir', f'{self.out_dir}',
                '--in-dir', f'{self.in_dir}',
                '--model-name', f'{self.model_name}']

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
            )
