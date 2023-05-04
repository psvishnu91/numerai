import configparser
import contextlib
import datetime
import functools
import gc
import hashlib
import json
import os
import os.path
import pickle
import re
import tempfile
import time
import typing
import urllib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import boto3
import flatdict
import lightgbm as lgb
import mlflow.entities
import numerapi
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection._split import indexable
from tqdm.notebook import tqdm

import mlflow

ERA_COL = "era"
TARGET_COL = "target_cyrus_v4_20"
DATA_TYPE_COL = "data_type"
EXAMPLE_PREDS_COL = "example_preds"

MODEL_FOLDER = "./data/models"
MODEL_CONFIGS_FOLDER = "model_configs"
PREDICTION_FILES_FOLDER = "prediction_files"

DF = pd.DataFrame
DF_STYLE = [
    # table properties
    dict(
        selector=" ",
        props=[
            ("margin", "0"),
            ("font-family", '"Helvetica", "Arial", sans-serif'),
            ("border-collapse", "collapse"),
            ("border", "none"),
            ("border", "2px solid #ccf"),
        ],
    ),
    # header color
    dict(selector="thead", props=[("background-color", "lightblue")]),
    # background shading
    dict(selector="tbody tr:nth-child(even)", props=[("background-color", "#fff")]),
    dict(selector="tbody tr:nth-child(odd)", props=[("background-color", "#eee")]),
    # cell spacing
    dict(
        selector="td",
        props=[
            ("padding", ".5em"),
            ("background-position-x", "initial"),
            ("background-position-y", "initial"),
        ],
    ),
    # header cell properties
    dict(selector="th", props=[("font-size", "100%"), ("text-align", "center")]),
]
########################################################################################
# Logging defaults utisl
########################################################################################


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
def hash_dict(d: Dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()[:6]


########################################################################################
# FILE UTILS
########################################################################################


def load_json(fl: str) -> Dict:
    with open(fl, "r") as f:
        return json.load(f)


def save_json(obj: Union[Dict, List], fl: str) -> None:
    with open(fl, "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def listdir_recursive(dir_path: str) -> List:
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


def save_model(model, name, model_folder=MODEL_FOLDER):
    try:
        Path(model_folder).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    pd.to_pickle(model, f"{model_folder}/{name}.pkl")


def load_model(name, model_folder=MODEL_FOLDER):
    path = Path(f"{model_folder}/{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{model_folder}/{name}.pkl")
    else:
        model = False
    return model


def pivot_df(df):
    """Given a df with two cols alpha and l1_ratio convert it table
    where rows are alpha and cols are l1_ratio.
    """
    return df.pivot(index="alpha", columns="l1_ratio", values="score")


########################################################################################
# AWS UTILS
########################################################################################
def download_from_s3_recursively(
    s3_path: str,
    local_path: str,
    aws_credential_fl: str,
    aws_profile: str = "default",
    overwrite: bool = False,
    dry_run: bool = False,
    flatten: bool = False,
    flname: Optional[str] = None,
) -> None:
    """Sample usage::

    download_from_s3_recursively(
        s3_path="s3://bucket/prefix/",
        local_path="/path/to/dir",
        aws_credential_fl="~/.aws/credentials",
    )
    :param flatten: If True, all files will be downloaded to local_path.
    :param flname: If passed only files with this name will be downloaded.
    """
    # download files in s3_path recursively to local_path
    boto_session = build_boto_session(
        aws_credential_fl=aws_credential_fl, aws_profile=aws_profile
    )
    s3 = boto_session.resource("s3")
    bucket = urllib.parse.urlparse(s3_path).netloc
    prefix = urllib.parse.urlparse(s3_path).path.lstrip("/")
    for obj in s3.Bucket(bucket).objects.filter(Prefix=prefix):
        if flname is not None and not obj.key.endswith(flname):
            continue
        s3_fl = f"s3://{bucket}/{obj.key}"
        if flatten:
            local_fl = os.path.join(local_path, os.path.basename(obj.key))
        else:
            local_fl = os.path.join(
                local_path, os.path.relpath(path=obj.key, start=prefix)
            )
        os.makedirs(os.path.dirname(local_fl), exist_ok=True)
        if dry_run:
            log.info(f"[DRYRUN] Would download {s3_fl} to {local_fl}")
        elif os.path.exists(local_fl) and not overwrite:
            log.info(
                f"Would have downloaded {s3_fl} to {local_fl}. "
                f"But {local_fl} exists. Will not download again ..."
            )
        else:
            log.info(f"Downloading {s3_fl} to {local_fl}")
            s3.meta.client.download_file(Bucket=bucket, Key=obj.key, Filename=local_fl)


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
        local_path="/path/to/dir/",
        aws_credential_fl="~/.aws/credentials",
    )
    :param local_file:
    """
    # download s3_path to local_path
    boto_session = build_boto_session(
        aws_credential_fl=aws_credential_fl, aws_profile=aws_profile
    )
    s3 = boto_session.resource("s3")
    bucket = urllib.parse.urlparse(s3_path).netloc
    s3_key = urllib.parse.urlparse(s3_path).path.lstrip("/")
    s3_fl = f"s3://{bucket}/{s3_key}"
    if os.path.isdir(local_path):
        local_fl = os.path.join(local_path, os.path.basename(s3_key))
    else:
        local_fl = local_path
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


def upload_s3_file(
    local_path: str,
    s3_path: str,
    aws_credential_fl: str,
    aws_profile: str = "default",
    dry_run: bool = False,
) -> None:
    """Uploads a local file to s3."""
    boto_session = build_boto_session(
        aws_credential_fl=aws_credential_fl, aws_profile=aws_profile
    )
    s3 = boto_session.resource("s3")
    bucket = urllib.parse.urlparse(s3_path).netloc
    s3_key = urllib.parse.urlparse(s3_path).path.lstrip("/")
    s3_fl = f"s3://{bucket}/{s3_key}"
    if dry_run:
        log.info(f"[DRYRUN] Would upload {local_path} to {s3_fl}")
    else:
        log.info(f"Uploading {local_path} to {s3_fl}")
        s3.meta.client.upload_file(Filename=local_path, Bucket=bucket, Key=s3_key)


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


def fmt_features(df: DF, features: List[str], int8: bool, impute: bool) -> DF:
    if impute:
        df.loc[:, features] = df.loc[:, features].fillna(
            df[features].median(skipna=True)
        )
    df[ERA_COL] = df[ERA_COL].astype(int)
    df.loc[:, features] = df.loc[:, features].astype(np.int8 if int8 else np.float32)
    return df


def cast_features(df: DF, int8: bool) -> DF:
    return df.astype(np.int8 if int8 else np.float32)


########################################################################################
# DATA LOADING
########################################################################################


def download_data(
    version: str, data_path: str, as_int8: bool, include_live: bool = True
) -> Dict[str, str]:
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
        downloaded_fl_map["live"] = download_live_dataset(
            data_path=data_path, version=version, dtype_suf=dtype_suf
        )
    return downloaded_fl_map


def download_live_dataset(version: str, data_path: str, dtype_suf: str) -> str:
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


def get_era_to_date(eras: List[int], offset_wks=12):
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
    features: List[str],
    read_cols: List[str],
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
) -> List[str]:
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
    return features + target_cols + [ERA_COL, DATA_TYPE_COL]


def subsample_every_nth_era(df: DF, n: int = 4) -> DF:
    if n == 1:
        return df
    every_nth_era = set(df[ERA_COL].unique()[::n])  # type: ignore
    return df[df[ERA_COL].isin(every_nth_era)]  # type: ignore


########################################################################################
# MODEL TRAINING
########################################################################################


def train_model(
    train_df: pd.DataFrame,
    features: List[str],
    target: str,
    train_data_ident: str,
    params: Dict,
    model_rootdir: Optional[str] = None,
    model_name: Optional[str] = None,
    model_obj: Optional[Any] = None,
):
    """Trains a model and saves it to disk.

    This function will check if a model with the same name already exists in the
    model_rootdir. If it does, it will not retrain the model. If it does not, it will
    train a new model and save it to disk.

    :param model_obj: If not passed we will create a lgbm model with
        params["lgbm_params"].
    """
    st_time = time.time()
    model_name = model_name or build_model_name(
        train_data_ident=train_data_ident,
        target=target,
        params=params,
    )
    if model_rootdir is not None:
        model_folder = os.path.join(model_rootdir, train_data_ident)
        log.info(f"Checking for existing model '{model_name}'")
        train_model = load_model(
            name=model_name,
            model_folder=model_folder,
        )
        if train_model:
            log.info(f"{model_name} found, will not retrain.")
            return train_model
    log.info(f"Creating new model...")
    if model_obj is None:
        train_model = lgb.LGBMRegressor(**params["lgbm_params"])
    else:
        train_model = model_obj
    # skip rows where target is NA
    non_na_index = train_df[target].dropna().index
    train_model.fit(
        X=cast_features(
            df=train_df.loc[non_na_index, features],
            int8=params["int8"],
        ),
        y=train_df.loc[non_na_index, target],
    )
    if model_rootdir is not None:
        log.info(f"saving new model: {model_name}")
        model_folder = os.path.join(model_rootdir, train_data_ident)
        save_model(
            model=train_model,
            name=model_name,
            model_folder=model_folder,
        )
    log.info(f"Training time: {(time.time() - st_time) / 60:.2f} minutes")
    return train_model


def build_model_name(train_data_ident, target, params, suffix="") -> str:
    """Creates a unique name for a model based on training
    data, target col and settings"""
    return f"{suffix}_{train_data_ident}_{target}_{hash_dict(params)}"


def predict(pred_df, model, parameters):
    return model.predict(X=cast_features(df=pred_df, int8=parameters["int8"]))


def get_pred_col(target):
    return f"pred_trained_on_{target}"


def log_mflow(
    run: mlflow.entities.Run,
    target: Optional[str] = None,
    params: Optional[Dict] = None,
    train_ident: Optional[str] = None,
    metrics_dict: Optional[Dict] = None,
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
    features: List,
    train_ident: str,
    targets: List[str],
    raw_predn_cols: List[str],
    model_rootdir: str,
    params: Dict,
    expt_id: str,
    suffix: str = "",
    models: Optional[Dict[str, object]] = None,
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
    features: List[str],
    train_ident: str,
    target: str,
    params: Dict,
    model_rootdir: Optional[str] = None,
    expt_id: Optional[str] = None,
    suffix: str = "",
) -> Dict[str, object]:
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
    if expt_id is not None:
        ctx = mlflow.start_run(run_name=f"{model_nm}", experiment_id=expt_id)
    else:
        ctx = contextlib.suppress()
    with ctx as run:
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
        validation_stats = validation_metrics(
            validation_data=test_df,
            pred_cols=[predn_col],
            target_col=TARGET_COL,
        )
        if run is not None:
            # Log metrics to mlflow
            log_mflow(
                metrics_dict=validation_stats.iloc[0].to_dict(),
                model=model,
                target=target,
                model_nm=model_nm,
                run=run,
                params=flatdict.FlatDict(params, delimiter="."),
            )
    return {
        "model": model,
        "model_name": model_nm,
        "predn_col": predn_col,
        "validation_stats": validation_stats,
        "run": run,
        "top_5_lgbm_feature_names": get_top_imp_feats(model, n=5),
    }


def get_top_imp_feats(model, n):
    """Gets the top n important feature names from a lgbm model."""
    return np.array(model.feature_name_)[
        np.argsort(model.feature_importances_)[-n:][::-1]
    ].tolist()


def fmt_metrics_df(metrics_df, add_bar=True):
    """Converts metrics columns into bar plots and formats as percentages.
    See https://pasteboard.co/79IY9Trd8M9C.png
    """
    styled_df = metrics_df.style.format("{:.2%}").set_table_styles(DF_STYLE)
    if add_bar:
        return styled_df.bar(color=["#d65f5f", "#5fba7d"], align="zero")
    else:
        return styled_df


def cross_validate(
    train_df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Dict,
    cv: int,
    train_ident: str = "",
    model_rootdir: Optional[str] = None,
    expt_id: Optional[str] = None,
    suffix: str = "",
    val_target: Optional[str] = None,
):
    """Cross validates a model for a single target.

    We use :func:`get_time_series_cross_val_splits` to get the cross
    validation split eras as train_test_zip = zip(train_splits, test_splits).
    Then we build a model for each train-test split and compute the metrics using
    `build_model_for_target`

    :param val_target: If not None, then we use the `target` used to train the model
    as the validation target. Using a different validation target is useful when
    we want to validate say a model trained on jerome is useful to predict on nomi.
    """
    val_target = val_target or target
    log.info(f"Validation target: `{val_target}`")
    cv_metric_dfs, cv_models = [None] * cv, [None] * cv
    for i, (train_split, test_split) in tqdm(
        enumerate(time_series_split(df=train_df, n_splits=cv)),  # type: ignore
        desc="Each cross validation split",
        total=cv,
    ):
        log.info(f"Building model for target={target}; {i+1}/{cv}")
        train_cv_df = train_df.iloc[train_split]
        test_cv_df = train_df.iloc[test_split]
        # Log train and test split info
        train_era_rng = (train_cv_df[ERA_COL].min(), train_cv_df[ERA_COL].max())
        test_era_rng = (test_cv_df[ERA_COL].min(), test_cv_df[ERA_COL].max())
        log.info(f"Train split: {train_cv_df.shape}, min, max era: {train_era_rng}")
        log.info(f"Test split: {test_cv_df.shape}, min, max era: {test_era_rng}")
        # Build model for target
        suffix_cv = suffix + f"_cv{i}"
        cv_models[i] = train_model(  # type: ignore
            train_df=train_cv_df,
            features=features,
            target=target,
            params=params,
            train_data_ident=train_ident + f"_cv{i}",
            model_rootdir=model_rootdir,
        )
        # Predict with model
        # add cross validation split or other ident to the prediction column name
        predn_col = f"{get_pred_col(target)}{suffix_cv}"
        test_cv_df[predn_col] = predict(
            pred_df=test_cv_df[features], model=cv_models[i], parameters=params
        )
        # Compute metrics
        cv_metric_dfs[i] = validation_metrics(  # type: ignore
            validation_data=test_cv_df,
            pred_cols=[predn_col],
            target_col=val_target,
        )
    # Aggregate metrics across cross validation splits
    metrics = to_cv_agg_df(cv_metric_dfs=cv_metric_dfs)
    retval = {"models": cv_models, "metrics": metrics}
    if expt_id is None:
        return retval
    else:
        # Log metrics to mlflow
        with mlflow.start_run(run_name=f"{suffix}_avg_cv", experiment_id=expt_id):
            mlflow.log_params({"train_ident": train_ident, "target": target, **params})
            mlflow.log_metrics(metrics.loc["cv_mean"].to_dict())
            # save formatted metrics to a temp file.
            with tempfile.NamedTemporaryFile(
                prefix="cv_metrics_df", suffix=".html"
            ) as tmp:
                fmt_metrics_df(metrics).to_html(tmp.name)
                mlflow.log_artifact(tmp.name)
        return retval


def to_cv_agg_df(cv_metric_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregates metrics across cross validation splits."""
    cv_val_df = pd.concat([mdf.transpose() for mdf in cv_metric_dfs])
    cv_mean = cv_val_df.mean(axis=0)
    cv_std = cv_val_df.std(axis=0)
    cv_std_err = cv_std / len(cv_val_df) ** 0.5
    cv_ci_low = cv_mean - 1.96 * cv_std_err
    cv_ci_high = cv_mean + 1.96 * cv_std_err
    cv_val_df.loc["cv_mean"] = cv_mean
    cv_val_df.loc["cv_low"] = cv_ci_low
    cv_val_df.loc["cv_high"] = cv_ci_high
    cv_val_df.loc["cv_std"] = cv_std
    return cv_val_df


########################################################################################
# Neutralisation & metrics


def neutralize(
    df,
    columns,
    neutralizers=None,
    proportion=1.0,
    normalize=True,
    era_col="era",
    verbose=False,
):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    if verbose:
        iterator = tqdm(unique_eras)
    else:
        iterator = unique_eras
    for u in iterator:
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values.astype(np.float32)
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method="ordinal") - 0.5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T.astype(np.float32)
        exposures = df_era[neutralizers].values.astype(np.float32)
        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures, rcond=1e-6).dot(scores)
        )
        scores /= scores.std(ddof=0)
        computed.append(scores)
    return pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index)


def validation_metrics(validation_data, pred_cols, target_col=TARGET_COL):
    """Compute validation metrics for a set of predictions.

    :returns: DataFrame with metrics for each prediction column.
    """
    validation_stats = pd.DataFrame()
    for pred_col in pred_cols:
        # Check the per-era correlations on the validation set (out of sample)
        validation_correlations = validation_data.groupby(ERA_COL).apply(
            lambda d: numerai_corr2(d[pred_col], d[target_col])
        )
        mean = validation_correlations.mean()
        # compute the stddev of correlation across the eras to compute the Sharpe ratio.
        std = validation_correlations.std(ddof=0)
        # Sharpe ratio of the average per-era correlation (t-statistic)
        sharpe = mean / std
        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe
    return validation_stats


def numerai_corr2(preds, target):
    # rank (keeping ties) then Gaussianize predictions to standardize prediction distributions
    ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)
    # make targets centered around 0. This assumes the targets have a mean of 0.5
    centered_target = target - 0.5
    # raise both preds and target to the power of 1.5 to accentuate the tails
    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
    # finally return the Pearson correlation
    return np.corrcoef(preds_p15, target_p15)[0, 1]


def time_series_split(df, n_splits, embargo=12):
    """Taken from https://forum.numer.ai/t/era-wise-time-series-cross-validation/791

    This function is a generator that yields (train_index, test_index) splits
    for time series data. We ensure that the test set is always after the
    train set. The indices are the row numbers of the original DataFrame.

    Usage::

        for train_ix, test_ix in time_series_split(df, n_splits=3):
            train_df, test_df = df.iloc[train_ix], df.iloc[test_ix]

    Sample input::

        df_fake = pd.DataFrame({"era": np.hstack([np.arange(0, 40, 4)])})
            era
        0    0
        1    4
        ...
        9   36

    Sample output::

        time_series_split(df_fake, n_splits=2, embargo=8)
        Train
        [0, 4, 8, 12, 16, 20, 24]
        Test
        [36]  # eras 28, 32 are embargoed out

        Train
        [0, 4, 8, 12]
        Test
        [24]  # eras 16, 20 are embargoed out
    """
    n_samples = df.shape[0]
    row_era = df[ERA_COL].astype(int)
    n_folds = n_splits + 1
    uniq_era = np.unique(row_era)  # type: ignore
    n_eras = len(uniq_era)
    if n_folds > n_eras:
        raise ValueError(
            f"Cannot have number of folds ={n_folds} greater than the number of "
            f"eras: {n_eras}."
        )
    indices = np.arange(n_samples)
    test_size = n_eras // n_folds
    test_starts = range(test_size + n_eras % n_folds, n_eras, test_size)
    test_starts = list(test_starts)[::-1]
    for test_start in test_starts:
        # The test eras that follow the train eras without embargo
        test_eras_opts = uniq_era[test_start : test_start + test_size]
        # Select the eras post the embargo
        test_eras = test_eras_opts[test_eras_opts > uniq_era[test_start - 1] + embargo]
        yield (
            indices[row_era.isin(uniq_era[:test_start])],  # type: ignore
            indices[row_era.isin(test_eras)],  # type: ignore
        )


########################################################################################
# PLOTTING TOOLS
########################################################################################


def refmt_predcols(cols):
    return [refmt_predcol(c) for c in cols]


def refmt_predcol(col):
    """Use regex to extract col name
    pred_target_arthur_v4_20_wt0.3_cv1 -> arthur_v4_20_wt0.3"""
    if "cv" in col:
        return re.search(r"pred_target_(\w+)_cv", col).group(1)
    else:
        return re.search(r"pred_target_(.*)", col).group(1)


def compare_models_with_baseline(
    df: DF,
    competitor_predcols: List[str],
    baseline_col: str,
    target_col: str,
    to_plot: bool = True,
    plot_erabinsz: int = 200,
    to_refmt_predcols: bool = False,
    figsize=(18, 8),
    era_col: str = ERA_COL,
    include_legend: bool = True,
) -> Tuple[Any, pd.DataFrame]:
    """Provides erawise summary plots describing how good competitor_predcols are
    against baseline_col. Also provides a summary table.

    Example output: https://gcdnb.pbrd.co/images/SXxbqHBIZRdS.png?o=1

    Given a df with competitor_predcols and a baseline predcol, plots the erawise
    correlation for each of predcol and the baseline col. Also plots the absolute
    improvement over the baseline col for each of the competitor_predcols.

    In the summary dataframe, the columns are:
    - model
    - corr2
    - sharpe
    - corr2_prop_increase
    - sharpe_prop_increase
    - #eras_better
    - #eras_better_proportion
    - worst_corr_era / baseline_worst (< 0 better)

    :param df: A dataframe containing era, predcols+baseline_col and target.
    :param competitor_predcols: A list of predcols to compare against baseline_col.
    :param baseline_col: The baseline predcol to compare against.
    :param target_col: The target column, ex: `target_cyrus_v4_20`.
    :param to_plot: Whether to plot the erawise correlation and improvement plots.
    :param plot_erabinsz: The number of eras to bin together for the plots.
    :param to_refmt_predcols: Whether to reformat the predcols using regex.
    :param figsize: The figsize for the plots.

    :return: A tuple of (fig, summary_df).

    Example usage::

        df = pd.DataFrame({
            "era": ["era1", "era1", "era2", "era2", "era3", "era3"],
            "pred_target_mdl1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "pred_target_mdl2": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "pred_target_baseline": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "target": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })
        fig, summary_df = compare_models_with_baseline(
            df,
            competitor_predcols=["pred_target_mdl1", "pred_target_mdl2"],
            baseline_col="pred_target_baseline",
            target_col="target",
            to_plot=True,
            plot_erabinsz=2,
            to_refmt_predcols=False,
            era_col="era",
        )
        fmt_metrics_df(
            summary_df[["corr2", "sharpe", "corr2_prop_increase", "sharpe_prop_increase"]]
        )
    """
    # Compute erawise correlation for each predcol and the baseline col
    era_mdl_corrs = [
        df.groupby(era_col).apply(lambda d: numerai_corr2(d[predcol], d[target_col]))
        for predcol in competitor_predcols
    ]
    # models, competitor_predcols corrs
    era_mdl_corr_df = pd.concat(era_mdl_corrs, axis=1)
    era_mdl_corr_df.columns = competitor_predcols
    bl_era_corr = df.groupby(era_col).apply(
        lambda d: numerai_corr2(d[baseline_col], d[target_col])
    )
    # Compute the absolute improvement ove baseline.
    blc_rescaled = (bl_era_corr + 1.0) / 2.0
    mc_rescaled_df = (era_mdl_corr_df + 1.0) / 2.0
    improvement_df = mc_rescaled_df.sub(blc_rescaled, axis=0)
    if to_plot:
        fig = _plot_era_improvment(
            improvement_df,
            erabinsz=plot_erabinsz,
            figsize=figsize,
            era_col=era_col,
            include_legend=include_legend,
        )
    else:
        fig = None
    return (
        fig,
        _build_comparison_summary(
            era_models_corr_df=era_mdl_corr_df,
            baseline_era_corr=bl_era_corr,
            baseline_col=baseline_col,
            to_refmt_predcols=to_refmt_predcols,
        ),
    )


@typing.no_type_check
def _plot_era_improvment(
    improvement_df: pd.DataFrame,
    era_col: str = ERA_COL,
    title: str = "Δcorr2 = corr_model - corr_baseline",
    xlabel: str = "Eras",
    ylabel: str = "Δcorr2",
    erabinsz: int = 200,
    figsize=(18, 8),
    axes_fontsize=16,
    annot_fontsize=16,
    title_fontsize=16,
    suptitle_fontsize=20,
    include_legend=True,
) -> Any:
    """Plots the abs improvement over the baseline col for each predcol with
    seaborn or ploty.

    :param improvement_df: A df where the rows are eras and the columns are the
        percentage improvement in corr vs the baseline col.
    :param w_plotly: If True, plots with plotly, else with matplotlib.
    """
    num_competitors = improvement_df.shape[1]
    # We will plot two subplots in each row unless there's just one competitor
    n_rows = num_competitors // 2 + num_competitors % 2
    n_cols = 2 if num_competitors > 1 else 1
    fig, _axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    axes = _axes.flatten() if num_competitors > 1 else [_axes]
    better_df = pct_better_by_erabin(improvement_df, binsz=erabinsz, era_col=era_col)
    # Times 1.1 to give the plot some breathing space
    ymin = [improvement_df.min().min() * 1.3] * 2
    ymax = [improvement_df.max().max() * 1.3] * 2
    colors = plt.cm.rainbow(np.linspace(0, 1, better_df.shape[0]))
    for ax, col in zip(axes, improvement_df.columns):
        ax2 = ax.twinx()
        # Scatter plot the percentage improvement over baseline
        sns.scatterplot(data=improvement_df[col], ax=ax, label="_nolegend_")
        ax.set_ylabel(ylabel, fontsize=axes_fontsize)
        ax.set_xlabel(xlabel, fontsize=axes_fontsize)
        # Plot a line plot of percentage columns better than baseline in each era bin
        sns.lineplot(
            x=[bn.mid for bn in better_df.index],
            y=better_df[col],
            ax=ax2,
            color="k",
        )
        ax2.set_ylabel(f"% {era_col.lower()} better", fontsize=axes_fontsize)
        ax2.set_ylim([0, better_df.max().max() * 1.2])
        # Flood fill regions and show how often the combined is better than baseline
        for i, bn in enumerate(better_df.index):
            ax.fill_between(
                x=[bn.left, bn.right],
                y1=ymin,
                y2=ymax,
                alpha=0.15,
                color=colors[i],
                label="_nolegend_",  # Don't show in legend
            )
            ax.annotate(
                f"%{era_col.lower()} better\n{better_df.loc[bn, col]:.0f}%",
                xy=[bn.left, ymin[0]],
                fontsize=annot_fontsize,
            )
        # Compute how many eras better than baseline in total
        pct_above = (improvement_df[col] > 0).mean()
        ax.set_title(
            col + f"\nPercent of {era_col.lower()} better: {pct_above:.0%}",
            fontsize=title_fontsize,
        )
        ax.hlines(
            y=0,
            xmin=improvement_df.index.min(),
            xmax=improvement_df.index.max(),
            color="r",
            linestyle="--",
            label="Baseline model",
        )
        if include_legend:
            ax2.legend(
                ["%Eras better than baseline in erabin"],
                loc="upper left",
            )
    plt.suptitle(title, fontsize=suptitle_fontsize)
    plt.tight_layout()
    return fig


def _build_comparison_summary(
    era_models_corr_df: pd.DataFrame,
    baseline_era_corr: pd.Series,
    baseline_col: str,
    to_refmt_predcols: bool = False,
) -> pd.DataFrame:
    """Builds a summary dataframe comparing the erawise correlation of each model
    against the baseline model.

    :param era_cmp_corr_df: A dataframe containing the erawise correlation of each
        model.
    :param baseline_era_corr: A series containing the erawise correlation of the
        baseline model.
    :returns: A dataframe containing the summary statistics as described in
        :func:`compare_predcols_with_baseline`.
    """
    # Number of eras better than baseline
    num_eras_better = era_models_corr_df.gt(baseline_era_corr, axis=0).sum(axis=0)
    prop_eras_better = num_eras_better / era_models_corr_df.shape[0]
    # Compute the ratio of the worst correlation of models vs baseline
    worst_corr_ratio = (era_models_corr_df.min(axis=0) + 1) / (
        baseline_era_corr.min() + 1
    )
    # Compute mean corr and sharpe for the models and baseline
    mdl_mean_corr = era_models_corr_df.mean(axis=0)
    mdl_sharpe = mdl_mean_corr / era_models_corr_df.std(axis=0, ddof=0)
    bl_mean_corr = baseline_era_corr.mean()
    bl_sharpe = bl_mean_corr / baseline_era_corr.std(ddof=0)
    # Compute the percentage improvement of corr and sharpe over the baseline col for
    # each predcol
    prop_imp_corr = (mdl_mean_corr - bl_mean_corr) / bl_mean_corr
    prop_imp_sharpe = (mdl_sharpe - bl_sharpe) / bl_sharpe
    # Build the summary dataframe
    if to_refmt_predcols:
        fmtd_predcols = refmt_predcols(era_models_corr_df.columns.to_list())
    else:
        fmtd_predcols = era_models_corr_df.columns.to_list()
    summary_df = pd.DataFrame(
        {
            "model": ([baseline_col] + fmtd_predcols),
            "corr2": [bl_mean_corr] + mdl_mean_corr.to_list(),
            "sharpe": [bl_sharpe] + mdl_sharpe.to_list(),
            "corr2_prop_increase": [0.0] + prop_imp_corr.to_list(),
            "sharpe_prop_increase": [0.0] + prop_imp_sharpe.to_list(),
            "#eras_better": [0] + num_eras_better.to_list(),
            "#eras_better_proportion": [0.0] + prop_eras_better.to_list(),
            "worst_corr_era / baseline_worst (< 0 better)": (
                [1.0] + worst_corr_ratio.to_list()
            ),
        }
    )
    return summary_df.set_index("model")


def pct_better_by_erabin(improvement_df, binsz=200, era_col=ERA_COL):
    """Group eras into bins of size of binsz and compute what fraction of eras in each
    bin are better than the baseline col.
    """
    # Don't mutate
    pi_df = improvement_df.copy()
    min_era, max_era = pi_df.index.min(), pi_df.index.max()
    bins = np.arange(min_era, max_era + binsz, binsz)
    pi_df[f"{era_col}_bin"] = pd.cut(pi_df.index, bins=bins)
    pi_df[pi_df.columns[:-1]] = pi_df[pi_df.columns[:-1]] > 0.0
    better_df = pi_df.groupby(f"{era_col}_bin").mean() * 100.0
    return better_df


def plot_erafill(
    bin_annot_map,
    ax=None,
    ymin=0,
    ymax=1,
    y_annotation=0.5,
    add_era_annotation=False,
    figsize=(8, 3),
    xpos="left",
    fontsize=15,
    **fill_kwargs,
):
    """Plots erabins and annotations in floodfill format.

    :param bin_annot_map: A dictionary mapping erabins to annotations
        bin_map = {(min_era, max_era): "<annotation>"}.
    :param ax: The matplotlib axis to plot on. If None, a new figure is created.
    :param ymin: The minimum y value of the floodfill.
    :param ymax: The maximum y value of the floodfill.
    :param y_annotation: The y value of the annotation.
    :param add_era_annotation: Whether to add the era range to the annotation.
    :param figsize: The size of the figure to create if `ax` is None.
    :param xpos: where should we annotate, `left` or `center`
    :param fontsize: The fontsize of the annotation.
    :param fill_kwargs: Additional kwargs to pass to `ax.fill_between`.

    :returns: The matplotlib axis.

    Usage::

        bin_annot_map = {(1, 500): "train", (500,800): "val", (800, 1100): "test"}
        plot_erafill(bin_annot_map, xpos="center", alpha=0.5)  # we overwite alpha
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.tab20c(np.linspace(0, 1, len(bin_annot_map)))
    ymin_list = [ymin] * 2
    ymax_list = [ymax] * 2
    for i, (bn, annot) in enumerate(bin_annot_map.items()):
        kwargs = {"alpha": 0.15, **fill_kwargs}
        ax.fill_between(
            x=bn,
            y1=ymin_list,  # type: ignore
            y2=ymax_list,  # type: ignore
            color=colors[i],
            **kwargs,
        )
        annot_x = bn[0] if xpos == "left" else bn[0] + (bn[1] - bn[0]) * 0.4
        _annot = f"{annot}\nEra:{bn}" if add_era_annotation else annot
        ax.annotate(f"{_annot}", xy=[annot_x, y_annotation], fontsize=fontsize)
    ax.set_xlabel("Eras", fontsize=fontsize)
    ax.set_xlim(0, max(bn[1] for bn in bin_annot_map))
    return ax
