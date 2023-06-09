import logging
import numpy as np
import pandas as pd
import scipy
import warnings

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class EnsembleNeutralModel:
    def __init__(
        self,
        models,
        neutralisation_cols="all",
        neutralisation_prop=0.5,
        ensembling_fn=None,
    ):
        self.models = models
        self.features = next(iter(models.values())).feature_name_
        self.neutralisation_cols = neutralisation_cols
        self.neutralisation_prop = neutralisation_prop
        self.ensembling_fn = ensembling_fn or np.mean

    def predict(self, df: pd.DataFrame):
        return predict_ensemble(
            df=df,
            models=self.models,
            features=self.features,
            ensembling_fn=self.ensembling_fn,
            neutralisation_cols=self.neutralisation_cols,
            neutralisation_prop=self.neutralisation_prop,
        )

# For backward compatability
MultiTargetNeutralModel = EnsembleNeutralModel
    
def predict_ensemble(
    df,
    models,
    features,
    ensembling_fn=np.mean,
    neutralisation_cols=None,
    neutralisation_prop=0.5,
):
    pred_cols = []
    df = df.copy()
    df[features] = to_int8(df[features])
    for model_nm, model in tqdm(
        models.items(), desc="Predicting for each model", total=len(models)
    ):
        pred_col = "pred_" + model_nm
        df[pred_col] = model.predict(df[features])
        pred_cols.append(pred_col)
    ensemble_col = "pred_ensemble"
    logger.info(f"Ensembling predictions with {ensembling_fn.__name__}(): {pred_cols}")
    df[ensemble_col] = ensembling_fn(df[pred_cols], axis=1)
    if neutralisation_cols is None:
        pred = df[[ensemble_col]]
    else:
        ntr_cols = features if neutralisation_cols == "all" else neutralisation_cols
        logger.info("Neutralising the predictions")
        pred = neutralize(
            df=df,
            columns=[ensemble_col],
            neutralizers=ntr_cols,
            proportion=neutralisation_prop,
            normalize=True,
            era_col="era",
            verbose=True,
        )
    logger.info("Taking the rank percent")
    return pred.rank(pct=True).rename(columns={ensemble_col: "prediction"})


def to_int8(df_feat):
    """In pickle uploads the default dataset is float32"""
    # check if any float32 values
    if sum(df_feat.values.ravel() == 0.5) > 0:
        return (df_feat * 4).astype(np.int8)
    else:
        return df_feat.astype(np.int8)
        
    

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


########################################################################################
# Specific models
########################################################################################

#--------------------------------------------------------------------------------------#
# ARGENTINA
#--------------------------------------------------------------------------------------#

def argentina_ensemble(df, cyrus_wt=0.7, **kwargs):
    """Given a dataframe with predictions from cyrus and non-cyrus models,
    averages predictions of cyrus and non-cyrus models separately and combines
    them as a weighted sum.
    """
    cyrus_cols = [c for c in df.columns if 'cyrus' in c]
    non_cyrus_cols = [c for c in df.columns if 'cyrus' not in c]
    return (
        (df[cyrus_cols].mean(axis=1) * cyrus_wt)
        + (df[non_cyrus_cols].mean(axis=1) * (1-cyrus_wt))
    )   