"""Logic to model numerai data.

Contains logic amongst other things to

- Load, sample and save data.
- Get feature sets.
"""
import gc
import json
import pickle

import numpy as np
import pandas as pd

from typing import Union


Array = Union[np.ndarray, pd.Series]


def sample_and_save_data(
    train_path: str,
    val_path: str,
    train_min_erano: int,
    only_4th_erano: bool,
    sampled_save_fl: str,
) -> dict[str, pd.DataFrame]:
    """Load training and validation data into dataframes, sample them and save them to
    disk in `sampled_save_fl`.
    
    See :func:`load_sampled_data` for output format.
    """
    split_dfs = load_train_val_dfs(
        train_path=train_path,
        val_path=val_path,
        train_min_erano=train_min_erano,
        only_4th_erano=only_4th_erano,
    )
    # Save the split dataframes to disk
    print("Saving sampled data to disk...")
    with open(sampled_save_fl, "wb") as f:
        pickle.dump(split_dfs, f)
    return split_dfs


def load_sampled_data(sampled_save_fl: str) -> dict[str, pd.DataFrame]:
    """Load sampled data from disk."""
    print("Loading sampled data from disk...")
    with open(sampled_save_fl, "rb") as f:
        split_dfs = pickle.load(f)
    return split_dfs


def load_train_val_dfs(
    train_path: str,
    val_path: str,
    train_min_erano: int,
    only_4th_erano: bool,
) -> dict[str, pd.DataFrame]:
    """Load training and validation data into dataframes.

    - Load only recent eras in training data (defined by `train_min_erano`). Validation
        begins after max(train_df.eras).
    - Sample eras such that later eras are chosen more often than earlier eras.
    - Sample every 4th era as the other eras have overlapping time periods. We don't
        have to do this but this can help get a more diverse dataset while still
        limiting memory.

    This function deletes the untrimmed df from memory and runs garbage collection.
    For a sample input like below we will need 1.5 gigs of RAM. The full dataset
    takes 15 gigs of RAM.

    Sample input::

        split_dfs = load_train_val_dfs(
            train_path='data/v4.1/train_int8.parquet',
            val_path='data/v4.1/validation_int8.parquet',
            train_min_erano=350,
        )

    Sample output::

        split_dfs = {
            "train": pd.DataFrame(),
            "val": pd.DataFrame(),
        }
    """
    split_dfs = {}
    for split, path in [
        ("train", train_path),
        ("val", val_path),
    ]:
        print(f"Loading {split} split...")
        df = pd.read_parquet(path)
        df["erano"] = df.era.astype(int)
        print(f"Number of original rows: \t{len(df):,}\n")
        print(df.info())
        print(f"\nDeleting rows before era from {split}: \t{train_min_erano}")
        df_sml = df[df.erano > train_min_erano]
        sampled_eras = _sample_eras(eras=df.erano)
        if only_4th_erano:
            # sample every 4th era to get a more diverse dataset
            chosen_eras = sampled_eras & set(
                range(train_min_erano, df.erano.max() + 1, 4)
            )
        else:
            chosen_eras = sampled_eras
        print(f"Number of chosen eras: {len(chosen_eras)}")
        df_sml = df_sml[df_sml.erano.isin(chosen_eras)]
        print("Summary about the chosen eras:")
        print(
            pd.Series(df_sml.erano).describe(
                percentiles=[0.05, 0.1, 0.25, 0.5, 0.75]
            )
        )
        del df
        gc.collect()
        print(f"Number of small rows: \t\t{len(df_sml):,}\n")
        df_sml.info()
        split_dfs[split] = df_sml
        print("\n\n")
    return split_dfs


def _sample_eras(eras: Array) -> set[int]:
    """Sample eras such that later eras are chosen more often than earlier eras.
    We cannot guarantee the number of eras chosen each time so the output can be of
    variable size.
    :returns: Chosen eras as a set.

    The histogram:
    https://github.com/psvishnu91/numerai/blob/main/assets/images/readme/sampling_eras.svg

    Sample calculation::
        
        eras_uniq = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        era_wts = np.array([ 1.,  3.,  5.,  8., 11., 15., 19., 23., 27., 32., 36.])
        # divide by max
        era_wts = array([0.03, 0.08, 0.14, 0.22, 0.31, 0.4 , 0.51, 0.62, 0.74, 0.87, 1.])
        # sample
        {16, 17, 19, 20}
    """
    # The higher the power the more recent eras are chosen. We can also use a
    # different power. We want to only weight based on the length between the smallest
    # and the largest era. We thus subtract the smallest era from all eras and add 1.
    eras_uniq = np.unique(eras)
    era_wts = (eras_uniq - np.min(eras_uniq) + 1) ** 1.5
    era_wts = era_wts / np.max(era_wts)  # type: ignore
    return set(eras_uniq[np.random.binomial(n=1, p=era_wts) > 0])


def get_features(df):
    return [f for f in df.columns if f.startswith("feature")]


def get_featureset(feature_json_fl, fset):
    with open(feature_json_fl) as f:
        return json.load(f)["feature_sets"][fset]
