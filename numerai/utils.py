import configparser
import datetime
import pickle
import time
import flatdict
import functools
import gc
import hashlib
import json
import logging
import os
import os.path
import urllib

import mlflow
import mlflow.entities
import lightgbm as lgb
from typing import Callable, Optional, Union

import boto3
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import nmr_utils
import numerapi

DF = pd.DataFrame


def print_and_write(msg, root_dir=""):
    log_date = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
    msg_full = f"{log_date} {msg}"
    with open(os.path.join(root_dir, "log.txt"), "a") as f:
        f.write(msg_full + "\n")
    print(msg_full)


# HACK: For some reason logging is not working in the notebook.
# log = logging.getLogger(__name__)
class Logger:
    def __init__(self, root_dir=""):
        self.info = self.debug = functools.partial(print_and_write, root_dir=root_dir)


log = Logger()


########################################################################################
# MISC UTILS
########################################################################################
def hash_dict(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()[:6]


########################################################################################
# FILE UTILS
########################################################################################


def load_json(fl: str) -> dict:
    with open(fl, "r") as f:
        return json.load(f)


def save_json(obj: Union[dict, list], fl: str) -> None:
    with open(fl, "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def listdir_recursive(dir_path: str) -> list:
    """Recursively list all files in a directory."""
    files = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            files.append(os.path.join(root, filename))
    return files


def pickle_obj(obj, fl):
    with open(fl, "wb") as f:
        pickle.dump(obj, f)


def unpickle_obj(fl):
    with open(fl, "rb") as f:
        return pickle.load(f)


########################################################################################
# AWS UTILS
########################################################################################
def download_s3_file(
    s3_path: str,
    local_path: str,
    aws_credential_fl: str,
    aws_profile: str = "default",
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Sample usage::

    download_s3_file(
        s3_path="s3://bucket/prefix/file",
        local_path="/path/to/file",
        aws_credential_fl="~/.aws/credentials",
    )
    """
    # download s3_path to local_path
    boto_session = build_boto_session(
        aws_credential_fl=aws_credential_fl, aws_profile=aws_profile
    )
    s3 = boto_session.resource("s3")
    bucket = urllib.parse.urlparse(s3_path).netloc
    s3_key = urllib.parse.urlparse(s3_path).path.lstrip("/")
    s3_fl = f"s3://{bucket}/{s3_key}"
    local_fl = os.path.join(local_path, os.path.basename(s3_key))
    if dry_run:
        log.info(f"[DRYRUN] Would download {s3_fl} to {local_fl}")
    elif os.path.exists(local_fl) and not overwrite:
        log.info(
            f"Would have downloaded {s3_fl} to {local_fl}. "
            f"But {local_fl} exists. Will not download again ..."
        )
    else:
        log.info(f"Downloading {s3_fl} to {local_fl}")
        s3.meta.client.download_file(Bucket=bucket, Key=s3_key, Filename=local_fl)


def log_something():
    log.info("Hello World")


def upload_to_s3_recursively(
    dir_path: str,
    s3_path: str,
    aws_credential_fl: str,
    aws_profile: str = "default",
    dry_run: bool = False,
) -> None:
    """Sample usage::

    upload_to_s3_recursively(
        dir_path="/path/to/dir",
        s3_path="s3://bucket/prefix/",
        aws_credential_fl="~/.aws/credentials",
    )
    """
    # upload files in dir_path recursively to s3_path
    boto_session = build_boto_session(
        aws_credential_fl=aws_credential_fl, aws_profile=aws_profile
    )
    s3 = boto_session.resource("s3")
    bucket = urllib.parse.urlparse(s3_path).netloc
    prefix = urllib.parse.urlparse(s3_path).path.lstrip("/")
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_key = os.path.join(
                prefix,
                os.path.relpath(path=local_path, start=dir_path),
            )
            if dry_run:
                log.info(
                    f"[DRYRUN] Would upload {local_path} to s3://{bucket}/{s3_key}"
                )
            else:
                log.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                s3.meta.client.upload_file(
                    Filename=local_path, Bucket=bucket, Key=s3_key
                )


def build_boto_session(
    aws_credential_fl: str, aws_profile: str = "default"
) -> boto3.Session:
    cfg = read_aws_config(aws_credential_fl=aws_credential_fl, profile=aws_profile)
    return boto3.Session(
        aws_access_key_id=cfg["aws_access_key_id"],
        aws_secret_access_key=cfg["aws_secret_access_key"],
        region_name=cfg["region_name"],
    )


def read_aws_config(
    aws_credential_fl: str, profile: str = "default"
) -> configparser.SectionProxy:
    """
    Sample usage::
        cfg = read_aws_config(aws_credential_fl="~/.aws/credentials")
        aws_key = cfg["aws_access_key_id"]
        aws_secret = cfg["aws_secret_access_key"]
    """
    log.debug(f"Loading aws credenitals from {aws_credential_fl}...")
    cfg_parser = configparser.ConfigParser()
    # aws credentials file is usually at ~/.aws/credentials and we need to expand `~`.
    cfg_parser.read(os.path.expanduser(aws_credential_fl))
    return cfg_parser[profile]


########################################################################################
# FEATURISATION
########################################################################################


def fmt_features(df: DF, features: list[str], int8: bool, impute: bool) -> DF:
    if impute:
        df.loc[:, features] = df.loc[:, features].fillna(
            df[features].median(skipna=True)
        )
    df.loc[:, features] = df.loc[:, features].astype(np.int8 if int8 else np.float32)
    return df


def cast_features(df: DF, int8: bool) -> DF:
    return df.astype(np.int8 if int8 else np.float32)


########################################################################################
# DATA LOADING
########################################################################################


def download_data(
    version: str, data_path: str, as_int8: bool, include_live: bool = True
) -> dict[str, str]:
    """Downloads training data.

    :version: Version of the data (ex: `'v4.1'`).
    :param data_path: where data is to be saved.
    :param as_int8: If set to true, we will download files in int8 format.
         if you remove the int8 suffix for each of these files, you'll
         get features between 0 and 1 as floats. int_8 files are much smaller...
         but are harder to work with because some packages don't like ints and
         the way NAs are encoded.
    Sample output::

        {
            "train": "/path/to/train.parquet",
            "test": "/path/to/validation.parquet",
            "features_json": "/path/to/features.json",
            "val_example": "/path/to/validation_example_preds.parquet",
            "live": "/path/to/live.parquet",
        }
    """
    # A mapping from to an easy name to the downloaded fl path
    downloaded_fl_map = {}
    napi = numerapi.NumerAPI()
    os.makedirs(data_path, exist_ok=True)
    print("Downloading dataset files...")
    dtype_suf = "_int8" if as_int8 else ""
    for fl_key, fl in [
        ("train", f"train{dtype_suf}.parquet"),
        ("test", f"validation{dtype_suf}.parquet"),
        ("features_json", "features.json"),
        ("val_example", "validation_example_preds.parquet"),
    ]:
        src = os.path.join(version, fl)
        dst = os.path.join(data_path, src)
        print(f"Downloading {src} to {dst}...")
        napi.download_dataset(filename=src, dest_path=dst)
        downloaded_fl_map[fl_key] = dst
    if include_live:
        downloaded_fl_map["live"] = _download_live_dataset(
            data_path=data_path, version=version, dtype_suf=dtype_suf
        )
    return downloaded_fl_map


def _download_live_dataset(version: str, data_path: str, dtype_suf: str) -> str:
    """Downloads the live dataset for the current round.
    :param dtype_suf: "_int8" or "".
    """
    # Tournament data changes every week so we specify the round in their name.
    napi = numerapi.NumerAPI()
    current_round = napi.get_current_round()
    print(f"Current round: {current_round}")
    live_src = f"{version}/live{dtype_suf}.parquet"
    live_dst = os.path.join(data_path, f"{current_round}/live{dtype_suf}.parquet")
    print(f"Downloading {live_src} to {live_dst}...")
    napi.download_dataset(filename=live_src, dest_path=live_dst)
    return live_dst


def sample_within_eras(df: DF, frac: float) -> DF:
    """Sample rows within each era."""
    return df.groupby("era").apply(lambda x: x.sample(frac=frac, random_state=42))


def get_era_to_date(eras: list[int], offset_wks=12):
    """Returns a mapping from era to the date associated with that era.

    Each era is a week long and the most recent era is offset weeks back. We can
    use this information to walk back in time and find the date associated with
    any era.
    """
    # Get the date associated with each era.
    era_to_date = {}
    for i, era in enumerate(reversed(eras)):
        era_to_date[era] = (
            datetime.datetime.today() - datetime.timedelta(weeks=i + offset_wks)
        ).strftime("%Y-%m-%d")
    return era_to_date


def download_fls(
    data_path,
    s3_path,
    s3_files,
    aws_credential_fl,
    int8=False,
):
    for s3_fl in s3_files:
        if s3_fl.endswith("json") or s3_fl.endswith("parquet"):
            s3_flnm = s3_fl
        else:
            int8_pfx = "_int8" if int8 else ""
            s3_flnm = f"{s3_fl}{int8_pfx}.parquet"
        log.info(f"Downloading: {s3_flnm}")
        download_s3_file(
            s3_path=os.path.join(s3_path, s3_flnm),
            local_path=data_path,
            aws_credential_fl=aws_credential_fl,
        )


def get_df_loader(
    data_path: str,
    features: list[str],
    read_cols: list[str],
    sample_every_nth: int,
    impute: bool,
    int8: bool,
) -> Callable[[str], DF]:
    def load_df(fl_name: str) -> DF:
        full_df = pd.read_parquet(
            os.path.join(data_path, fl_name),
            columns=read_cols,
        )
        df = fmt_features(
            df=subsample_every_nth_era(df=full_df, n=sample_every_nth),
            features=features,
            int8=int8,
            impute=impute,
        )
        gc.collect()
        return df

    return load_df


def build_cols_to_read(
    feature_json_fl: str,
    feature_set_name: Optional[str] = None,
) -> list[str]:
    """Builds a list of columns to read from the downloaded data."""
    # read the feature metadata and get a feature set (or all the features)
    with open(feature_json_fl, "r") as f:
        feature_metadata = json.load(f)
    if feature_set_name:
        features = feature_metadata["feature_sets"][feature_set_name]
    else:
        features = list(feature_metadata["feature_stats"])
    target_cols = feature_metadata["targets"]
    # read in just those features along with era and target columns
    return features + target_cols + [nmr_utils.ERA_COL, nmr_utils.DATA_TYPE_COL]


def subsample_every_nth_era(df: DF, n: int = 4) -> DF:
    if n == 1:
        return df
    every_nth_era = set(df[nmr_utils.ERA_COL].unique()[::n])  # type: ignore
    return df[df[nmr_utils.ERA_COL].isin(every_nth_era)]  # type: ignore


########################################################################################
# MODEL TRAINING
########################################################################################


def train_model(
    train_df: pd.DataFrame,
    model_rootdir: str,
    features: list[str],
    target: str,
    train_data_ident: str,
    params: dict,
    model_name: Optional[str] = None,
    overwrite=False,
) -> lgb.LGBMRegressor:
    """Trains a model and saves it to disk.

    This function will check if a model with the same name already exists in the
    model_rootdir. If it does, it will not retrain the model. If it does not, it will
    train a new model and save it to disk.
    """
    st_time = time.time()
    model_name = model_name or build_model_name(
        train_data_ident=train_data_ident,
        target=target,
        params=params,
    )
    model_folder = os.path.join(model_rootdir, train_data_ident)
    if not overwrite:
        log.info(f"Checking for existing model '{model_name}'")
        train_model = nmr_utils.load_model(
            name=model_name,
            model_folder=model_folder,
        )
        if train_model:
            log.info(f"{model_name} found, will not retrain.")
            return train_model
    log.info(f"Creating new model...")
    train_model = lgb.LGBMRegressor(**params["lgbm_params"])
    # skip rows where target is NA
    non_na_index = train_df[target].dropna().index
    train_model.fit(
        X=cast_features(
            df=train_df.loc[non_na_index, features],
            int8=params["int8"],
        ),
        y=train_df.loc[non_na_index, target],
    )
    log.info(f"saving new model: {model_name}")
    nmr_utils.save_model(
        model=train_model,
        name=model_name,
        model_folder=model_folder,
    )
    log.info(f"Training time: {(time.time() - st_time) / 60:.2f} minutes")
    return train_model


def build_model_name(train_data_ident, target, params, suffix="") -> str:
    """Creates a unique name for a model based on training
    data, target col and settings"""
    return f"{train_data_ident}_{target}{suffix}_{hash_dict(params)}"


def predict(pred_df, model, parameters):
    return model.predict(X=cast_features(df=pred_df, int8=parameters["int8"]))


def get_pred_col(target):
    return f"pred_trained_on_{target}"


def log_mflow(
    run: mlflow.entities.Run,
    target: Optional[str] = None,
    params: Optional[dict] = None,
    train_ident: Optional[str] = None,
    metrics_dict: Optional[dict] = None,
    model: Optional[object] = None,
    model_nm: Optional[str] = None,
):
    params = params or {}
    mlflow.log_params(params={"train_ident": train_ident, "target": target, **params})
    if metrics_dict:
        mlflow.log_metrics(metrics_dict)
    if model:
        mlflow.lightgbm.log_model(model, artifact_path=model_nm)
        model_uri = f"runs:/{run.info.run_id}/lightgbm-model"
        mv = mlflow.register_model(model_uri=model_uri, name=model_nm)
        log.info(f"Logged model={mv.name}, version={mv.version}")


def group_eras_into_bins(df, era_col, bin_col_nm, biz_sz):
    """Groups eras into bins of biz_sz eras each."""
    eras = df[era_col].unique()
    era_bins = np.array_split(eras, len(eras) / biz_sz)
    era_bin_map = {era: i for i, era_bin in enumerate(era_bins) for era in era_bin}
    df[bin_col_nm] = df[era_col].map(era_bin_map)
    return df


def build_models_for_all_targets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list,
    train_ident: str,
    targets: list[str],
    raw_predn_cols: list[str],
    model_rootdir: str,
    params: dict,
    expt_id: str,
    suffix: str = "",
    models: Optional[dict[str, object]] = None,
) -> None:
    """Builds a model for each target in targets.

    This function will build a model for each target in targets. It will then
    save the model to disk and log the model to mlflow.

    models is map from target to the models that are built.
    """
    for i, target in tqdm(
        enumerate(targets, start=1), desc="Each target", total=len(targets)
    ):
        log.info(f"Building model for target={target}; {i}/{len(targets)}")
        model_build_result = build_model_for_target(
            train_df=train_df,
            test_df=test_df,
            features=features,
            train_ident=train_ident,
            target=target,
            model_rootdir=model_rootdir,
            params=params,
            expt_id=expt_id,
            suffix=suffix,
        )
        if models is not None:
            models[target] = model_build_result["model"]
        raw_predn_cols.append(model_build_result["predn_col"])  # type: ignore


def build_model_for_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    train_ident: str,
    target: str,
    model_rootdir: str,
    params: dict,
    expt_id: str,
    suffix: str = "",
) -> dict[str, object]:
    """Builds a model for a single target.

    This function will build a model for a single target. It will then
    save the model to disk and log the model to mlflow. The predictions
    will be added to the test_df as a column with the name of the target and
    the suffix. The model will be added to the models dict.

    HACK: Mutates test_df.

    :param suffix: This is suffixed to the prediction column name and the model name.
        Useful when building model for different xval splits (suffix = "_cv{i}").
    """
    # Train model
    model_nm = build_model_name(
        train_data_ident=train_ident,
        target=target,
        params=params,
        suffix=suffix,
    )
    run = mlflow.start_run(run_name=model_nm, experiment_id=expt_id)
    model = train_model(
        train_df=train_df,
        features=features,
        target=target,
        params=params,
        train_data_ident=train_ident,
        model_name=model_nm,
        model_rootdir=model_rootdir,
    )
    # Predict with model
    # add cross validation split or other ident to the prediction column name
    predn_col = f"{get_pred_col(target)}{suffix}"
    test_df[predn_col] = predict(
        pred_df=test_df[features], model=model, parameters=params
    )
    # Compute metrics
    validation_stats = nmr_utils.validation_metrics(
        validation_data=test_df,
        pred_cols=[predn_col],
        example_col=None,
        fast_mode=True,
        target_col=nmr_utils.TARGET_COL,
        include_mmc=False,
    )
    # Log metrics to mlflow
    log_mflow(
        metrics_dict=validation_stats.iloc[0].to_dict(),
        model=model,
        target=target,
        model_nm=model_nm,
        run=run,
        params=flatdict.FlatDict(params, delimiter="."),
    )
    mlflow.end_run()
    return {
        "model": model,
        "model_name": model_nm,
        "predn_col": predn_col,
        "validation_stats": validation_stats,
        "run": run,
    }


def fmt_metrics_df(metrics_df):
    """Converts metrics columns into bar plots and formats as percentages.
    See https://pasteboard.co/79IY9Trd8M9C.png
    """
    return metrics_df.style.bar(color=["#d65f5f", "#5fba7d"], align="zero").format(
        "{:.2%}"
    )
