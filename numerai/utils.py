import configparser
import datetime
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
import nmr_utils

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
):
    """Trains a model and saves it to disk.

    This function will check if a model with the same name already exists in the
    model_rootdir. If it does, it will not retrain the model. If it does not, it will
    train a new model and save it to disk.
    """
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
        verbose=True,
    )
    log.info(f"saving new model: {model_name}")
    nmr_utils.save_model(
        model=train_model,
        name=model_name,
        model_folder=model_folder,
    )
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
