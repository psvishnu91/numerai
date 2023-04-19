"""Script to download data from the Numerai API.

This script downloads the latest data from the Numerai API and
splits into various train, validation and test sets. Optionally
uploads them to s3.

.. note:: We will treat the numerai validation dataset as test.

If running on AWS EC2, use ``r5a.xlarge`` or larger.

The files are downloaded as below::

    # metadata file contains the min, max eras of each split along with num rows.
    data_path/splits_folder/metadata.json
    data_path/splits_folder/train_p0-60{_int8}.parquet
    data_path/splits_folder/val_p60-80.parquet
    data_path/splits_folder/train_p0-100_test_p0-80.parquet

Usage::

    python split_numerai_dataset.py \
        --version v4.1 \
        --data-path data/ \
        --s3-path s3://numerai/dataset/ \
        --aws-credentials ~/.aws/credentials
    
    # If you are rerunning the script for a new split
    python split_numerai_dataset.py \
        --version v4.1 \
        --data-path data/ \
        --s3-path s3://numerai/dataset/ \
        --aws-credentials ~/.aws/credentials \
        --upload-splits-only \
        --labels traintest_p80-100 
"""
import argparse
import dataclasses as dc
import datetime
import gc
import logging
import os.path
from typing import Any, Optional

import nmr_utils as nu
import pandas as pd
import tqdm
import utils as ut

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SPLITS_FOLDER = "splits"
#: A mapping of which data splits we would like to generate. Source can be either
#: "train", "test" or "combined". The final label will be
#: `{label_prefix}_p{min_era_percent}-{max_era_percent}.parquet`. If ``label`` is
#: specified, it will be used as the final label. The splits will contain the eras
#: from min_er_percent (inclusive) to max_era_percent (exclusive).
SPLIT_CONFIG = [
    {
        "source": "train",
        "label_prefix": "train",
        "min_era_percent": 0.0,
        "max_era_percent": 60.0,
    },
    {
        "source": "train",
        "label_prefix": "train",
        "min_era_percent": 0.0,
        "max_era_percent": 80.0,
    },
    {
        "source": "train",
        "label_prefix": "val",
        "min_era_percent": 60.0,
        "max_era_percent": 80.0,
    },
    {
        "source": "train",
        "label_prefix": "val",
        "min_era_percent": 80.0,
        "max_era_percent": 100.0,
    },
    {
        "source": "test",
        "label_prefix": "test",
        "min_era_percent": 0.0,
        "max_era_percent": 80.0,
    },
    {
        "source": "test",
        "label_prefix": "test",
        "min_era_percent": 80.0,
        "max_era_percent": 100.0,
    },
    {
        "source": "combined",
        "label_prefix": "traintest",
        "min_era_percent": 0.0,
        "max_era_percent": 80.0,
    },
    {
        "source": "combined",
        "label_prefix": "traintest",
        "min_era_percent": 80.0,
        "max_era_percent": 100.0,
    },
    {
        "source": "combined",
        "label_prefix": "traintest",
        "min_era_percent": 0.0,
        "max_era_percent": 100.0,
    },
]


@dc.dataclass(frozen=True)
class Datasets:
    #: Concatenated train and test data.
    data: pd.DataFrame
    train_eras: pd.Series
    test_eras: pd.Series
    all_eras: pd.Series
    training_indices: pd.Index
    test_indices: pd.Index
    all_indices: pd.Index


def main():
    opts = _parse_opts()
    log.info("Downloading data...")
    downloaded_fl_map = ut.download_data(
        version=opts.version,
        data_path=opts.data_path,
        as_int8=opts.int8,
        include_live=False,
    )
    print(f"Downloaded file to path map: {downloaded_fl_map}")
    datasets = _load_datasets(downloaded_fl_map)
    gc.collect()
    _split_and_save_datasets(
        datasets=datasets,
        version=opts.version,
        data_path=opts.data_path,
        split_config=SPLIT_CONFIG,
        as_int8=opts.int8,
        splits_folder=SPLITS_FOLDER,
        labels=opts.labels,
    )
    log.info("Uploading to s3...")
    if opts.upload_splits_only:
        dir_path = os.path.join(opts.data_path, opts.version, SPLITS_FOLDER)
        upload_s3_path = os.path.join(opts.s3_path, opts.version, SPLITS_FOLDER)
    else:
        dir_path = opts.data_path
        upload_s3_path = opts.s3_path
    ut.upload_to_s3_recursively(
        dir_path=dir_path,
        s3_path=upload_s3_path,
        aws_credential_fl=opts.aws_credentials,
        aws_profile=opts.aws_profile,
        dry_run=opts.s3_dry_run,
    )


def _parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        default="v4.1",
        help="Version of the data to download. Default: v4.1",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help=(
            "If set, we will download files in int8 format. If you remove the int8 "
            "suffix for each of these files, you'll get features between 0 and 1 as "
            "floats. int_8 files are much smaller... but are harder to work with "
            "because some packages don't like ints and the way NAs are encoded."
        ),
        default=False,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to save the downloaded data to. Default: data/",
    )
    parser.add_argument(
        "--s3-path",
        type=str,
        help=(
            "If set, uploads the data to s3 at this path. Example: "
            "`s3://numerai-datasets/2023/04/17/v4.1/`."
        ),
        required=True,
    )
    parser.add_argument(
        "--s3-dry-run",
        action="store_true",
        help="If set, will not upload to s3.",
        default=False,
    )
    parser.add_argument(
        "--aws-credentials",
        type=str,
        help=(
            "Path to a file containing the aws credentials. If not set, will look "
            "for the file at ~/.aws/credentials."
        ),
        required=True,
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        help=(
            "If set, will use this profile in the aws credentials file. If not set, "
            "will use the default profile."
        ),
        default="default",
    )
    parser.add_argument(
        "--upload-splits-only",
        action="store_true",
        help=(
            "If set, will only upload the splits to s3. This is useful if you've "
            "already uploaded the data to s3 and just want to update the splits."
        ),
        default=False,
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        type=str,
        help=(
            "If set, will only compute and upload the splits for the given labels. "
            "Example: `--labels val_p60-80`. This is useful if you've already "
            "computed and uploaded some splits."
        ),
        default=None,
    )
    return parser.parse_args()


def _load_datasets(downloaded_fl_map) -> Datasets:
    """Load the downloaded data and combine train and test into one dataframe."""
    training_data = pd.read_parquet(downloaded_fl_map["train"])
    test_data = pd.read_parquet(downloaded_fl_map["test"])
    all_data = pd.concat([training_data, test_data])
    all_data[nu.ERA_COL] = all_data[nu.ERA_COL].astype(int)
    return Datasets(
        data=all_data,
        train_eras=pd.Series(training_data[nu.ERA_COL].astype(int).unique()),
        test_eras=pd.Series(test_data[nu.ERA_COL].astype(int).unique()),
        all_eras=pd.Series(all_data[nu.ERA_COL].astype(int).unique()),
        # save indices for easier data selection later
        training_indices=training_data.index,
        test_indices=test_data.index,
        all_indices=all_data.index,
    )


def _split_and_save_datasets(
    datasets: Datasets,
    version: str,
    data_path: str,
    split_config: list[dict[str, str]],
    as_int8: bool,
    splits_folder: str,
    labels: Optional[set[str]] = None,
):
    """Split the data into train and test into various splits according to
    ``split_config`` and save to disk. See SPLIT_CONFIG definition for more details.

    Output will be a folder structure like::

        data_path/splits_folder/metadata.json
        data_path/splits_folder/train_p0-60{_int8}.parquet
        data_path/splits_folder/val_p60-80.parquet
        data_path/splits_folder/train_p0-100_test_p0-80.parquet
    """
    metadata = {}
    os.makedirs(os.path.join(data_path, version, splits_folder), exist_ok=True)
    log.info(f"Creating and saving splits ...")
    for split in tqdm.tqdm(split_config):
        split_fl = _build_split_flname(split=split, as_int8=as_int8)
        # If `labels` is given, only files that start with these labels will be computed.
        if (labels is not None) and not any(
            split_fl.startswith(label) for label in labels
        ):
            continue
        split_data = _split_data(datasets=datasets, split=split)
        split_df = split_data["split_df"]
        metadata[split_fl] = {
            "min_era": split_data["min_era"],
            "max_era": split_data["max_era"],
            "num_rows": len(split_df),
        }
        split_df.to_parquet(os.path.join(data_path, version, splits_folder, split_fl))
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ut.save_json(
        obj=metadata,
        fl=os.path.join(data_path, version, splits_folder, f"metadata_{ts}.json"),
    )


def _build_split_flname(split: dict[str, str], as_int8: bool) -> str:
    try:
        split_label = split["label"]
    except KeyError:
        split_label = (
            f"{split['label_prefix']}"
            f"_p{int(split['min_era_percent'])}-{int(split['max_era_percent'])}"
        )
    if as_int8:
        split_file_name = f"{split_label}_int8.parquet"
    else:
        split_file_name = f"{split_label}.parquet"
    return split_file_name


def _split_data(datasets: Datasets, split: dict[str, Any]) -> dict:
    """Split the data according to the split config."""
    if split["source"] == "train":
        split_df = datasets.data.loc[datasets.training_indices]
        eras = datasets.train_eras
    elif split["source"] == "test":
        split_df = datasets.data.loc[datasets.test_indices]
        eras = datasets.test_eras
    elif split["source"] == "combined":
        split_df = datasets.data
        eras = datasets.all_eras
    else:
        raise ValueError(f"Unknown split source: {split['source']}")
    min_era = round(eras.quantile(split["min_era_percent"] / 100.0))
    max_era = round(eras.quantile(split["max_era_percent"] / 100.0))
    split_df = split_df[
        (split_df[nu.ERA_COL] >= min_era) & (split_df[nu.ERA_COL] < max_era)
    ]
    return {"split_df": split_df, "min_era": min_era, "max_era": max_era}


if __name__ == "__main__":
    main()
