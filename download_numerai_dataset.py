"""Script to download data from the Numerai API.

This script downloads the latest data from the Numerai API and
stores it in the data folder. 

It stores them in the following folder structure

    data/v4.1/train_int8.parquet
    data/v4.1/validation_int8.parquet
    data/444/v4.1/live_int8.parquet
    data/v4.1/features.json

Usage::
    
    python download_numerai_dataset.py --round 444 --version v4.1
"""
import numerapi
import pathlib
import argparse


def _parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Round to download data for. If not specified, "
        "the current round is downloaded.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v4.1",
        help="Version of the data to download. Default: v4.1",
    )
    return parser.parse_args()


def download_data(napi, cur_round=None, version="v4.1"):
    """Downloads training, validation and tournament-live data."""
    fl_to_path = {}
    cur_round = cur_round or napi.get_current_round()
    pathlib.Path(f"data/{cur_round}").mkdir(exist_ok=True, parents=True)
    fl_to_downpath = {
        f"{version}/live_int8.parquet": f"data/{cur_round}/{version}/live_int8.parquet",
        f"{version}/train_int8.parquet": f"data/{version}/train_int8.parquet",
        f"{version}/validation_int8.parquet": f"data/{version}/validation_int8.parquet",
        f"{version}/features.json": f"data/{version}/features.json",
    }
    print(f"Current round: \t{cur_round}\n")
    print(f"Download file to paths: \n\t{fl_to_downpath}\n")
    print(f"Downloading...\n")
    for filename, dest_path in fl_to_downpath.items():
        napi.download_dataset(filename=filename, dest_path=dest_path)
    downd_fls = "\n\t".join(str(pt) for pt in pathlib.Path("data").rglob("*.*"))
    print(f"Verify folders and files within the created data folder: \n\t{downd_fls}")
    return cur_round, fl_to_downpath


if __name__ == "__main__":
    opts = _parse_opts()
    napi = numerapi.NumerAPI()
    download_data(napi, cur_round=opts.round, version=opts.version)
