import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "santander-customer-transaction-prediction"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ID_COL = "ID_code"
TARGET = "target"
SEED = 42
GPU_PLATFORM_ID = 1
GPU_DEVICE_ID = 0


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        log(f"START: {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        log(f"END: {self.name}, elapsed={time.perf_counter() - self.start:.2f}s")


def print_environment():
    """打印训练环境，确认 GPU 设备。"""
    log("========== Environment ==========")
    log(f"Python: {sys.version.replace(chr(10), ' ')}")
    log(f"Python executable: {sys.executable}")
    log(f"Platform: {platform.platform()}")
    log(f"LightGBM version: {lgb.__version__}")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
    log("nvidia-smi stdout:")
    print(result.stdout, flush=True)
    if result.stderr:
        log("nvidia-smi stderr:")
        print(result.stderr, flush=True)


def check_lightgbm_gpu_support():
    """用小数据强制跑 GPU；失败就报错，不回退 CPU。"""
    log("========== LightGBM GPU Support Check ==========")
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame(rng.normal(size=(3000, 20)).astype("float32"))
    y = pd.Series(rng.integers(0, 2, size=3000))
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 30,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "device_type": "gpu",
        "device": "gpu",
        "gpu_platform_id": GPU_PLATFORM_ID,
        "gpu_device_id": GPU_DEVICE_ID,
        "max_bin": 255,
        "n_jobs": -1,
        "verbose": 1,
    }
    with Timer("LightGBM GPU smoke test"):
        model = lgb.LGBMClassifier(**params)
        model.fit(x, y, eval_set=[(x, y)], eval_metric="auc", callbacks=[lgb.log_evaluation(10)])
    log("LightGBM GPU smoke test passed.")


def detect_real_test_rows(test, features):
    """识别 Santander test 中的 real rows：每行在某些匿名变量上会有 test 内唯一值。"""
    with Timer("detect real/synthetic test rows"):
        unique_counts = np.zeros(len(test), dtype=np.int16)
        for i, col in enumerate(features, 1):
            if i % 25 == 0:
                log(f"real-row detection progress: {i}/{len(features)}")
            vc = test[col].value_counts(dropna=False)
            unique_values = set(vc.index[vc == 1])
            unique_counts += test[col].isin(unique_values).values.astype(np.int16)
        real_mask = unique_counts > 0
        log(f"real test rows={int(real_mask.sum())}, synthetic rows={int((~real_mask).sum())}")
        log(
            "unique-per-row stats: "
            f"min={unique_counts.min()}, max={unique_counts.max()}, "
            f"mean={unique_counts.mean():.4f}, std={unique_counts.std():.4f}"
        )
        return real_mask, unique_counts


def build_features(train, test):
    """构造可解释特征：real-test-aware 频次、唯一性、rank、行级统计。"""
    with Timer("feature construction"):
        y = train[TARGET].astype(int).copy()
        features = [c for c in train.columns if c.startswith("var_")]
        tr = train[features].copy()
        te = test[features].copy()

        real_mask, test_unique_counts = detect_real_test_rows(test, features)
        reference = pd.concat([train[features], test.loc[real_mask, features]], axis=0, ignore_index=True)

        log("building row-level statistics")
        for df in (tr, te):
            df["row_mean"] = df[features].mean(axis=1)
            df["row_std"] = df[features].std(axis=1)
            df["row_min"] = df[features].min(axis=1)
            df["row_max"] = df[features].max(axis=1)
            df["row_sum"] = df[features].sum(axis=1)
            df["row_skew"] = df[features].skew(axis=1)
            df["row_kurt"] = df[features].kurtosis(axis=1)

        log("building real-test-aware count and unique features")
        for i, col in enumerate(features, 1):
            if i % 25 == 0:
                log(f"count/unique progress: {i}/{len(features)}")
            vc = reference[col].value_counts(dropna=False)
            tr_count = tr[col].map(vc).fillna(0).astype("float32")
            te_count = te[col].map(vc).fillna(0).astype("float32")
            tr[f"{col}_count"] = tr_count
            te[f"{col}_count"] = te_count
            tr[f"{col}_unique"] = (tr_count == 1).astype("int8")
            te[f"{col}_unique"] = (te_count == 1).astype("int8")

        count_cols = [f"{c}_count" for c in features]
        unique_cols = [f"{c}_unique" for c in features]
        for df in (tr, te):
            df["count_mean"] = df[count_cols].mean(axis=1)
            df["count_std"] = df[count_cols].std(axis=1)
            df["count_min"] = df[count_cols].min(axis=1)
            df["count_max"] = df[count_cols].max(axis=1)
            df["unique_sum"] = df[unique_cols].sum(axis=1)
        te["test_unique_sum"] = test_unique_counts.astype("float32")
        tr["test_unique_sum"] = 0.0

        log("building rank features on train + detected real test reference")
        all_for_rank = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
        for i, col in enumerate(features, 1):
            if i % 25 == 0:
                log(f"rank progress: {i}/{len(features)}")
            ranks = rankdata(all_for_rank[col].values, method="average").astype("float32")
            ranks /= ranks.max()
            tr[f"{col}_rank"] = ranks[: len(tr)]
            te[f"{col}_rank"] = ranks[len(tr):]

        tr = tr.astype("float32")
        te = te.astype("float32")
        log(f"constructed features: train={tr.shape}, test={te.shape}")
        return tr, te, y, real_mask


def fit_lgbm_gpu(train_x, test_x, y):
    """5 折 GPU LightGBM 训练；失败直接停止。"""
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 7000,
        "learning_rate": 0.008,
        "num_leaves": 13,
        "max_depth": -1,
        "min_child_samples": 60,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.10,
        "reg_alpha": 1.5,
        "reg_lambda": 6.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 1,
        "device_type": "gpu",
        "device": "gpu",
        "gpu_platform_id": GPU_PLATFORM_ID,
        "gpu_device_id": GPU_DEVICE_ID,
        "max_bin": 255,
    }

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_x), dtype=np.float64)
    pred = np.zeros(len(test_x), dtype=np.float64)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(train_x, y), 1):
        log(f"========== GPU fold {fold}/5 ==========")
        with Timer(f"LightGBM GPU fold {fold} training"):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                train_x.iloc[tr_idx],
                y.iloc[tr_idx],
                eval_set=[(train_x.iloc[va_idx], y.iloc[va_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(500), lgb.log_evaluation(250)],
            )
        va_pred = model.predict_proba(train_x.iloc[va_idx], num_iteration=model.best_iteration_)[:, 1]
        te_pred = model.predict_proba(test_x, num_iteration=model.best_iteration_)[:, 1]
        auc = roc_auc_score(y.iloc[va_idx], va_pred)
        log(f"fold {fold}: AUC={auc:.6f}, best_iter={model.best_iteration_}, device=gpu")
        oof[va_idx] = va_pred
        pred += te_pred / 5
        scores.append({"fold": fold, "auc": float(auc), "best_iteration": int(model.best_iteration_), "device": "gpu"})

    oof_auc = roc_auc_score(y, oof)
    log(f"GPU OOF AUC={oof_auc:.6f}")
    return oof, pred, oof_auc, scores, params


def main():
    total_start = time.perf_counter()
    print_environment()
    check_lightgbm_gpu_support()

    with Timer("read data"):
        train = pd.read_csv(DATA_DIR / "train.csv")
        test = pd.read_csv(DATA_DIR / "test.csv")
        sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    train_x, test_x, y, real_mask = build_features(train, test)

    with Timer("GPU training"):
        oof, pred, oof_auc, scores, params = fit_lgbm_gpu(train_x, test_x, y)

    oof_path = OUTPUT_DIR / "oof_lgbm_magic_real_gpu.csv"
    pred_path = OUTPUT_DIR / "pred_lgbm_magic_real_gpu.csv"
    sub_path = OUTPUT_DIR / "submission_lgbm_magic_real_gpu.csv"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: oof}).to_csv(oof_path, index=False)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred}).to_csv(pred_path, index=False)
    sub = sample.copy()
    sub[TARGET] = pred
    sub.to_csv(sub_path, index=False)

    summary = {
        "mode": "lgbm_magic_real_gpu",
        "lightgbm_version": lgb.__version__,
        "python": sys.version,
        "rows": int(len(train)),
        "n_features": int(train_x.shape[1]),
        "real_test_rows": int(real_mask.sum()),
        "synthetic_test_rows": int((~real_mask).sum()),
        "oof_auc": float(oof_auc),
        "fold_scores": scores,
        "params": params,
        "gpu_platform_id": GPU_PLATFORM_ID,
        "gpu_device_id": GPU_DEVICE_ID,
        "notes": "使用 Santander real/synthetic test 识别构造频次特征；LightGBM 指定 NVIDIA OpenCL GPU，不回退 CPU。",
    }
    (OUTPUT_DIR / "lgbm_magic_real_gpu_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(f"saved: {oof_path}")
    log(f"saved: {pred_path}")
    log(f"saved: {sub_path}")
    print(sub[TARGET].describe(), flush=True)
    log(f"TOTAL elapsed={time.perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()
