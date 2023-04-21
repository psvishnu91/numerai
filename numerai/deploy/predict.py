"""This script is run daily to upload live predictions to Numerai.

This script will download the most recent round's live data, make predictions
using the most recent model, and upload the predictions to Numerai. All models
use the int8 features.

Usage::

    python predict.py --model-name=<model_name>

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
import numerapi
import pandas as pd

logging.basicConfig(filename="log.txt", filemode="a")

DATA_VERSION = "v4.1"
ERA_COL = "era"
DATA_TYPE_COL = "data_type"
TARGET_COL = "target_nomi_v4_20"
MODEL_DIR = "./deployed_models/"
PREDICTIONS_PATH = "predictions.csv"

DEFAULT_PUBLIC_ID = None
DEFAULT_SECRET_KEY = None

napi = numerapi.NumerAPI(
    public_id=os.getenv("NUMERAI_PUBLIC_ID", DEFAULT_PUBLIC_ID),
    secret_key=os.getenv("NUMERAI_SECRET_KEY", DEFAULT_SECRET_KEY),
)


def main():
    opts = _parse_opts()
    model_id = opts.model_id
    model_path = os.path.join(MODEL_DIR, opts.model_id + ".pkl")
    logging.info(f"Loading model from {model_id} from {model_path}...")
    wrapped_model = unpickle_obj(fl=model_path)
    predictions = predict(napi=napi, wrapped_model=wrapped_model)
    submit(predictions=predictions, model_id=model_id)


def _parse_opts():
    parser = argparse.ArgumentParser(description="Numerai live prediction script")
    parser.add_argument(
        "--model-id",
        type=str,
        help="Model ID to use for predictions",
        default=os.getenv("MODEL_ID"),
    )
    return parser.parse_args()


def unpickle_obj(fl):
    with open(fl, "rb") as f:
        return pickle.load(f)


def predict(napi, wrapped_model):
    logging.info("reading prediction data")
    napi.download_dataset(f"{DATA_VERSION}/live.parquet")
    predict_data = pd.read_parquet(
        f"{DATA_VERSION}/live.parquet",
        columns=wrapped_model.features,
    )
    logging.info("generating predictions")
    predictions = wrapped_model.predict(
        predict_data.filter(like="feature_", axis="columns")
    )
    predictions = pd.DataFrame(
        predictions, columns=["prediction"], index=predict_data.index
    )
    return predictions


def submit(predictions, model_id, predict_output_path=PREDICTIONS_PATH):
    logging.info("writing predictions to file and submitting")
    predictions.to_csv(predict_output_path)
    napi.upload_predictions(predict_output_path, model_id=model_id)


if __name__ == "__main__":
    main()
