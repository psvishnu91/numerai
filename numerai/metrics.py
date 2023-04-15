"""Module contains metrics for analysing Numerai models.

Loads of code in this file are directly lifted from the Numerai forums.
"""
import csv
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn import metrics, model_selection
from sklearn.model_selection import _split as sk_split
from xgboost import XGBRegressor


########################################################################################
#                           Xval logic
# Lifted from https://forum.numer.ai/t/era-wise-time-series-cross-validation/791
########################################################################################

class TimeSeriesSplitGroups(sk_split._BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X: Iterable, y: pd.Series, groups: pd.Series):
        X, y, groups = sk_split.indexable(X, y, groups)  # type: ignore
        n_samples = len(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                f"Cannot have number of folds ={n_folds} greater than the number "
                f"of samples: {n_groups}."
            )
        indices = np.arange(n_samples)
        test_size = n_groups // n_folds
        test_starts = range(test_size + n_groups % n_folds, n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            yield (
                indices[groups.isin(group_list[:test_start])],
                indices[groups.isin(group_list[test_start : test_start + test_size])],
            )


def numerai_score(y_true, y_pred, eras):
    """Models should be scored based on rank-correlation (spearman) with target."""
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]


def correlation_score(y_true, y_pred):
    """Convenient while working to evaluate based on the regular (pearson) correlation"""
    return np.corrcoef(y_true, y_pred)[0,1]


def xval_example(train_df):
    """
    """
    features = [f for f in train_df.columns if f.startswith("feature")]
    target = "target_kazutsugi"
    train_df["erano"] = train_df.era.str.slice(3).astype(int)
    eras = train_df.erano
    cv_score = []
    models = []
    for lr in [0.006, 0.008, 0.01, 0.012, 0.014]:
        for cs in [0.06, 0.08, 0.1, 0.12, 0.14]:
            for md in [4, 5, 6]:
                models.append(
                    XGBRegressor(
                        colsample_bytree=cs,
                        learning_rate=lr,
                        n_estimators=2000,
                        max_depth=md,
                        nthread=8,
                    )
                )
    for model in models:
        score = np.mean(
            model_selection.cross_val_score(
                model,
                train_df[features],
                train_df[target],
                cv=TimeSeriesSplitGroups(5),
                n_jobs=1,
                groups=eras,
                scoring=metrics.make_scorer(numerai_score, greater_is_better=True),
            )
        )
        cv_score.append(score)
    print(cv_score)
