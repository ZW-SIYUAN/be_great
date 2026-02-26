"""
shared/eval_core.py
===================
纯计算函数：分类器定义、ML 效用评估、判别器、DCR。
不含任何路径、数据集名称或 I/O 逻辑。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone as sklearn_clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── 默认超参 ──────────────────────────────────────────────────────────────────
SEEDS: List[int] = [42, 1, 2, 3, 4]
DISC_TEST_SIZE: float = 0.30


# ── 数据预处理 ────────────────────────────────────────────────────────────────

def encode_categoricals(
    dfs: List[pd.DataFrame],
    cat_cols: List[str],
) -> Dict[str, LabelEncoder]:
    """
    用所有 dfs 的并集对 cat_cols 原地做 LabelEncoder。
    返回 {col: encoder}，供解码或分布图使用。
    target 列可以包含在 cat_cols 中。
    """
    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([d[col] for d in dfs]).astype(str)
        le.fit(combined)
        for d in dfs:
            d[col] = le.transform(d[col].astype(str))
        encoders[col] = le
    return encoders


# ── 分类器 ────────────────────────────────────────────────────────────────────

def get_classifiers(seed: int) -> Dict:
    """返回 {name: sklearn estimator}，3 个标准下游分类器。"""
    return {
        "LR": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed,
                                     n_jobs=-1),
    }


# ── ML 效用 ───────────────────────────────────────────────────────────────────

def run_ml_trial(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    syn_dict: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    target: str,
    seed: int,
) -> Dict:
    """
    单次 ML 效用评估（一个 seed）。
    返回 {model_name: {split_name: {acc, f1, auc}}}
    split_name 包含 'original' 和 syn_dict 的所有 key。
    """
    X_te = test_real[feature_cols].values
    y_te = test_real[target].values

    splits = {"original": (train_real[feature_cols].values,
                           train_real[target].values)}
    splits.update({name: (df[feature_cols].values, df[target].values)
                   for name, df in syn_dict.items()})

    results = {}
    for model_name, clf in get_classifiers(seed).items():
        row = {}
        for split_name, (Xtr, ytr) in splits.items():
            c = sklearn_clone(clf)
            c.fit(Xtr, ytr)
            yp  = c.predict(X_te)
            ypr = c.predict_proba(X_te)[:, 1]
            row[split_name] = {
                "acc": accuracy_score(y_te, yp) * 100,
                "f1":  f1_score(y_te, yp, average="weighted",
                                zero_division=0) * 100,
                "auc": roc_auc_score(y_te, ypr) * 100,
            }
        results[model_name] = row
    return results


def compute_ml_tables(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    syn_dict: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    target: str,
    seeds: List[int] = SEEDS,
) -> Dict:
    """
    多 seed 聚合 ML 效用。
    返回 {model: {split: {metric: [per_seed_values]}}}
    """
    models     = ["LR", "DT", "RF"]
    split_keys = ["original"] + list(syn_dict.keys())
    metrics    = ["acc", "f1", "auc"]

    store = {m: {s: {k: [] for k in metrics} for s in split_keys}
             for m in models}

    for seed in seeds:
        trial = run_ml_trial(train_real, test_real, syn_dict,
                             feature_cols, target, seed)
        for m in models:
            for s in split_keys:
                for k in metrics:
                    store[m][s][k].append(trial[m][s][k])
    return store


# ── 判别器 ────────────────────────────────────────────────────────────────────

def run_discriminator_trial(
    train_real: pd.DataFrame,
    syn_df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    test_size: float = DISC_TEST_SIZE,
) -> float:
    """
    RF 判别器：区分真实训练数据 (1) 与合成数据 (0)。
    返回 hold-out 准确率 (%)。越低越好，50% = 无法区分。
    """
    X = np.vstack([train_real[feature_cols].values,
                   syn_df[feature_cols].values])
    y = np.array([1] * len(train_real) + [0] * len(syn_df))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te)) * 100


def compute_discriminator(
    train_real: pd.DataFrame,
    syn_dict: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    seeds: List[int] = SEEDS,
) -> Dict[str, List[float]]:
    """返回 {method_name: [per_seed_accuracy]}"""
    return {name: [run_discriminator_trial(train_real, df, feature_cols, s)
                   for s in seeds]
            for name, df in syn_dict.items()}


# ── DCR ───────────────────────────────────────────────────────────────────────

def compute_dcr(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    syn_dict: Dict[str, pd.DataFrame],
    feature_cols: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    计算 Distance to Closest Record（L2，min-max 归一化）。
    返回 (dcr_test, {method_name: dcr_array})
    dcr_test 是测试集相对训练集的基准距离。
    """
    X_train = train_real[feature_cols].values.astype(float)
    X_test  = test_real[feature_cols].values.astype(float)

    col_min = X_train.min(0)
    col_max = X_train.max(0)
    rng     = np.where(col_max > col_min, col_max - col_min, 1.0)

    Xtr_n    = (X_train - col_min) / rng
    Xte_n    = (X_test  - col_min) / rng
    dcr_test = cdist(Xte_n, Xtr_n, metric="euclidean").min(axis=1)

    dcr_syn: Dict[str, np.ndarray] = {}
    for name, df in syn_dict.items():
        Xs_n = (df[feature_cols].values.astype(float) - col_min) / rng
        dcr_syn[name] = cdist(Xs_n, Xtr_n, metric="euclidean").min(axis=1)

    return dcr_test, dcr_syn
