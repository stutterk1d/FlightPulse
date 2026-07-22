import numpy as np

from src.flightpulse.features import build_features
from src.flightpulse.preprocess import build_preprocessor


def test_preprocessor_fits_and_transforms(raw_df):
    feats = build_features(raw_df)
    X, y = feats.drop(columns=["label"]), feats["label"].to_numpy()
    pre = build_preprocessor()
    Xt = pre.fit_transform(X, y)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > 0
    assert np.isfinite(np.asarray(Xt, dtype=float)).all()


def test_unseen_categories_do_not_raise(raw_df):
    feats = build_features(raw_df)
    X, y = feats.drop(columns=["label"]), feats["label"].to_numpy()
    pre = build_preprocessor()
    pre.fit(X, y)

    unseen = X.head(5).copy()
    unseen["origin"] = "ZZZ"          # airport never seen in training
    unseen["dest"] = "QQQ"
    unseen["route"] = "ZZZ-QQQ"
    unseen["airline"] = "XX"

    Xt = pre.transform(unseen)        # must not raise
    assert Xt.shape[0] == 5


def test_transform_output_width_is_stable(raw_df):
    feats = build_features(raw_df)
    X, y = feats.drop(columns=["label"]), feats["label"].to_numpy()
    pre = build_preprocessor()
    Xt_fit = pre.fit_transform(X, y)
    Xt_new = pre.transform(X.head(10))
    assert Xt_fit.shape[1] == Xt_new.shape[1]
