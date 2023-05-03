"""This script is run daily to upload live predictions to Numerai.

This script will download the most recent round's live data, make predictions
using the most recent model, and upload the predictions to Numerai. All models
use the int8 features.

Usage::

    python predict.py --model-name=<model_name> --model-id=<model_id> --dryrun

Supported models:
1. `nomi_v4_20`
2. `six_targets_avg_n50`
3. `skip_lwimp_six_targets_avg_n50`
"""
import argparse
import os
import os.path
import logging
import pickle
import sys
import numerapi
import pandas as pd
# HACK: import all the functions from deploy_model.py
from deploy_model import *

file_handler = logging.FileHandler(filename="predict.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)

DATA_VERSION = "v4.1"
ERA_COL = "era"
DATA_TYPE_COL = "data_type"
TARGET_COL = "target_nomi_v4_20"
MODEL_DIR = "./models/"
PREDICTIONS_PATH = "predictions.csv"

DEFAULT_PUBLIC_ID = None
DEFAULT_SECRET_KEY = None

napi = numerapi.NumerAPI(
    public_id=os.getenv("NUMERAI_PUBLIC_ID", DEFAULT_PUBLIC_ID),
    secret_key=os.getenv("NUMERAI_SECRET_KEY", DEFAULT_SECRET_KEY),
)


def main():
    opts = _parse_opts()
    model_name = opts.model_name
    model_path = os.path.join(MODEL_DIR, model_name + ".pkl")
    logger.info(f"Loading model from {model_name} from {model_path}...")
    wrapped_model = unpickle_obj(fl=model_path)
    predictions = predict(napi=napi, wrapped_model=wrapped_model)
    logger.info(predictions)
    if opts.dryrun:
        logger.info("Dry run, not uploading predictions")
    else:
        submit(predictions=predictions, model_id=opts.model_id)


def _parse_opts():
    parser = argparse.ArgumentParser(description="Numerai live prediction script")
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use for predictions",
        required=True,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="Model id. Get this from the numer.ai models page.",
        required=True,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run, do not upload predictions",
        default=False,
    )
    return parser.parse_args()


def unpickle_obj(fl):
    with open(fl, "rb") as f:
        return pickle.load(f)


def predict(napi, wrapped_model):
    logger.info("reading prediction data")
    current_round = napi.get_current_round()
    dest_fl = f"{DATA_VERSION}/live_{current_round}.parquet"
    napi.download_dataset(filename=f"{DATA_VERSION}/live.parquet", dest_path=dest_fl)
    logger.info(f"Downloaded live data to {dest_fl}...")
    predict_data = pd.read_parquet(dest_fl)
    logger.info("generating predictions")
    return wrapped_model.predict(predict_data)


def submit(predictions, model_id, predict_output_path=PREDICTIONS_PATH):
    logger.info("writing predictions to file and submitting")
    predictions.to_csv(predict_output_path)
    napi.upload_predictions(predict_output_path, model_id=model_id)


if __name__ == "__main__":
    main()
