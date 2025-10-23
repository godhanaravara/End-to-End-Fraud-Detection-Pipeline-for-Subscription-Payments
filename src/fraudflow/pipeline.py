from __future__ import annotations
import logging
from pathlib import Path
from pyspark.sql import DataFrame
from .config import settings
from .spark import SparkSessionManager
from .io import DataIngestor
from .transform import PaymentTransformer
from .features import FeatureBuilder
from .model import LightGBMTrainer, FraudScorer

logger = logging.getLogger(__name__)

FEATURES = [
    # base
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrig", "delta_bal",
    "oldbalanceDest", "newbalanceDest", "delta_bal_dest",
    "is_weekend",
    # flags
    "dest_zero_before", "dest_zero_after",
    "amount_close_to_delta_orig", "amount_close_to_delta_dest",
    # ratios/logs
    "log_amount", "amt_over_old_org", "amt_over_old_dest",
    # one-hots
    "type_PAYMENT", "type_TRANSFER", "type_CASH_OUT", "type_CASH_IN", "type_DEBIT",
    # velocities (origin)
    "tx_count_orig_1d", "tx_count_orig_7d", "sum_amount_orig_7d", "max_amount_orig_30d",
    # velocities (dest)
    "tx_count_dest_1d", "tx_count_dest_7d", "sum_amount_dest_7d", "max_amount_dest_30d",
]

class FraudPipeline:
    def __init__(self) -> None:
        self.spark_mgr = SparkSessionManager()
        self.ingestor = DataIngestor(self.spark_mgr)
        self.transformer = PaymentTransformer()
        self.features = FeatureBuilder()

    def extract(self, csv_path: Path | None = None) -> DataFrame:
        path = csv_path or settings.csv_path
        df = self.ingestor.read_csv(path)
        self.ingestor.write_parquet(df, settings.bronze_dir)
        return df

    def transform(self, bronze_df: DataFrame | None = None) -> DataFrame:
        if bronze_df is None:
            bronze_df = self.spark_mgr.spark.read.parquet(str(settings.bronze_dir))
        silver = self.transformer.to_silver(bronze_df).coalesce(2) 
        self.ingestor.write_parquet(silver, settings.silver_dir)
        return silver

    def build_features(self, silver_df: DataFrame | None = None) -> DataFrame:
        if silver_df is None:
            silver_df = self.spark_mgr.spark.read.parquet(str(settings.silver_dir))
        feats = self.features.build(silver_df)
        return feats

    def train(self, feats_df: DataFrame) -> dict:
        trainer = LightGBMTrainer(target="isFraud")
        metrics = trainer.fit(feats_df, FEATURES, limit=1_000_000)
        trainer.save(settings.model_path)
        return metrics

    def score(self, feats_df: DataFrame) -> None:
        scorer = FraudScorer(settings.model_path)
    
        # keeping isFraud so the report can compute ROC/PR
        keep_cols = [c for c in ["step", "amount", "isFraud"] if c in feats_df.columns and c not in FEATURES]
        sel_cols = FEATURES + keep_cols
    
        scored_sdf = scorer.score_spark(feats_df.select(*sel_cols), FEATURES)
    
        (
            scored_sdf
            .repartition(64, "step")
            .write
            .mode("overwrite")
            .partitionBy("step")
            .parquet(str(settings.gold_dir.joinpath("scores.parquet")))
        )
        logger.info(f"Wrote scores to {settings.gold_dir}/scores.parquet/")

