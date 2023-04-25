"""Copied verbatim from https://github.com/numerai/example-scripts/blob/91de57/utils.py
"""
import numpy as np
import pandas as pd
import scipy
from halo import Halo
from scipy import stats
from tqdm import tqdm
from pathlib import Path
import json
from scipy.stats import skew

ERA_COL = "era"
TARGET_COL = "target_nomi_v4_20"
DATA_TYPE_COL = "data_type"
EXAMPLE_PREDS_COL = "example_preds"

spinner = Halo(text="", spinner="dots")

MODEL_FOLDER = "./data/models"
MODEL_CONFIGS_FOLDER = "model_configs"
PREDICTION_FILES_FOLDER = "prediction_files"


def save_prediction(df, name):
    try:
        Path(PREDICTION_FILES_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    df.to_csv(f"{PREDICTION_FILES_FOLDER}/{name}.csv", index=True)


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


def save_model_config(model_config, model_name):
    try:
        Path(MODEL_CONFIGS_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    with open(f"{MODEL_CONFIGS_FOLDER}/{model_name}.json", "w") as fp:
        json.dump(model_config, fp)


def load_model_config(model_name):
    path_str = f"{MODEL_CONFIGS_FOLDER}/{model_name}.json"
    path = Path(path_str)
    if path.is_file():
        with open(path_str, "r") as fp:
            model_config = json.load(fp)
    else:
        model_config = False
    return model_config


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[: len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2 :]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


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


def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1))
    )

    correction = proportion * (
        exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0])
    )
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


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


def validation_metrics(validation_data, pred_cols, target_col=TARGET_COL):
    validation_stats = pd.DataFrame()
    for pred_col in pred_cols:
        # Check the per-era correlations on the validation set (out of sample)
        validation_correlations = validation_data.groupby(ERA_COL).apply(
            lambda d: numerai_corr2(d[pred_col], d[target_col])
        )
        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std
        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe
    return validation_stats

