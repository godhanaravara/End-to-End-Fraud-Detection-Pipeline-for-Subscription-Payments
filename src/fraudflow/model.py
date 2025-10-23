from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


# Helpers
def _precision_recall_at_rate(y_true: pd.Series, y_score: pd.Series, rate: float) -> tuple[float, float]:
    """
    Compute precision / recall at top-k rate (e.g., 0.001 = top 0.1%).
    """
    n = len(y_score)
    k = max(int(np.ceil(rate * n)), 1)
    order = np.argsort(-y_score.values)
    topk_idx = order[:k]
    y_top = y_true.values[topk_idx]
    tp = y_top.sum()
    prec = float(tp / k)
    rec = float(tp / max(y_true.sum(), 1))
    return prec, rec


# Trainer
class LightGBMTrainer:
    """
    Time-aware split, class imbalance handling, track PR@budgets.
    """

    def __init__(self, target: str = "isFraud") -> None:
        self.target = target
        self.model = LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.05,
            num_leaves=255,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )

    def _to_pandas(self, df: DataFrame, cols: list[str], limit: int | None) -> pd.DataFrame:
        sdf = df.select(*cols)
        if limit:
            sdf = sdf.limit(limit)
        return sdf.toPandas()

    def _time_valid_split(self, pdf: pd.DataFrame, step_col: str = "step", valid_frac: float = 0.1):
        cutoff = np.quantile(pdf[step_col].values, 1.0 - valid_frac)
        train_pdf = pdf[pdf[step_col] <= cutoff].copy()
        valid_pdf = pdf[pdf[step_col] > cutoff].copy()
        return train_pdf, valid_pdf

    def fit(self, df: DataFrame, features: list[str], limit: int | None = 1_000_000) -> dict:
        cols = features + [self.target]
        pdf = self._to_pandas(df, cols, limit)

        train_pdf, valid_pdf = self._time_valid_split(pdf, step_col="step", valid_frac=0.1)
        X_tr, y_tr = train_pdf[features], train_pdf[self.target].astype(int)
        X_va, y_va = valid_pdf[features], valid_pdf[self.target].astype(int)

        neg = int((y_tr == 0).sum()); pos = int((y_tr == 1).sum())
        spw = float(neg / max(pos, 1))
        self.model.set_params(scale_pos_weight=spw)

        logger.info("Training LightGBM…")

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=["auc", "average_precision"],
        )

        tr_score = self.model.predict_proba(X_tr)[:, 1]
        va_score = self.model.predict_proba(X_va)[:, 1]

        from sklearn.metrics import roc_auc_score, average_precision_score
        train_roc = float(roc_auc_score(y_tr, tr_score))
        train_pr = float(average_precision_score(y_tr, tr_score))
        valid_roc = float(roc_auc_score(y_va, va_score))
        valid_pr = float(average_precision_score(y_va, va_score))

        # precision/recall@k helpers
        import numpy as np
        def _pr_at(y_true, y_score, rate):
            n = len(y_score); k = max(int(np.ceil(rate * n)), 1)
            order = np.argsort(-y_score); top = y_true.values[order[:k]]
            tp = top.sum(); prec = float(tp / k); rec = float(tp / max(y_true.sum(), 1))
            return prec, rec

        p01, r01 = _pr_at(y_va, va_score, 0.001)
        p05, r05 = _pr_at(y_va, va_score, 0.005)
        p10, r10 = _pr_at(y_va, va_score, 0.010)

        metrics = {
            "train_roc_auc": train_roc, "train_pr_auc": train_pr,
            "valid_roc_auc": valid_roc, "valid_pr_auc": valid_pr,
            "valid_precision_at_0.1pct": p01, "valid_recall_at_0.1pct": r01,
            "valid_precision_at_0.5pct": p05, "valid_recall_at_0.5pct": r05,
            "valid_precision_at_1pct": p10, "valid_recall_at_1pct": r10,
            "scale_pos_weight": spw,
        }
        logger.info(f"Metrics: {metrics}")
        return metrics

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved model to {path}")


# Spark-native scorer
class FraudScorer:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)

    def score(self, df: DataFrame, features: list[str], limit: int | None = None) -> pd.DataFrame:
        """(Legacy) Collects to driver. Prefer score_spark for big data."""
        pdf = df.select(*features).toPandas() if limit is None else df.select(*features).limit(limit).toPandas()
        model = joblib.load(self.model_path)
        pdf["score"] = model.predict_proba(pdf[features])[:, 1]
        return pdf

    def score_spark(self, df: DataFrame, features: list[str]) -> DataFrame:
        """
        Distributed scoring without collecting to driver.
        Uses mapInPandas + a broadcasted model blob to score each Arrow batch.
        """
        spark = df.sparkSession
        sc = spark.sparkContext

        # Raw model byte
        model_bytes = sc.broadcast(self.model_path.read_bytes())

        def _score_iter(pdf_iter: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            # lazy-load per task
            model = joblib.load(io.BytesIO(model_bytes.value))
            for pdf in pdf_iter:
                local = pdf.copy()
                local["score"] = model.predict_proba(local[features])[:, 1]
                yield local

        # Output schema = input schema + score: double
        out_schema = df.schema.add("score", DoubleType())
        # Repartition to keep Arrow batches moderate
        df_scored = df.repartition(200).mapInPandas(_score_iter, schema=out_schema)
        return df_scored
