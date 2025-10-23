from __future__ import annotations
import logging
from pyspark.sql import DataFrame, Window, functions as F

logger = logging.getLogger(__name__)

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]

class FeatureBuilder:
    def build(self, silver_df: DataFrame) -> DataFrame:
        """
        Builds an enriched feature set:
          - Orig + Dest balances and deltas
          - Consistency flags (orig/dest)
          - Ratios/logs
          - Type one-hots
          - Velocity features by nameOrig/nameDest over 1d/7d/30d windows
        """
        logger.info("Building ML features…")
        df = silver_df

        # Destination-side deltas & flags
        df = df.withColumn("delta_bal_dest", F.col("newbalanceDest") - F.col("oldbalanceDest"))
        df = df.withColumn("dest_zero_before", (F.col("oldbalanceDest") == 0).cast("int"))
        df = df.withColumn("dest_zero_after",  (F.col("newbalanceDest") == 0).cast("int"))

        # Consistency flags (orig & dest)
        eps = F.lit(1.0)
        df = df.withColumn(
            "amount_close_to_delta_orig",
            (F.abs(F.col("amount") - (F.col("newbalanceOrig") - F.col("oldbalanceOrg"))) <= eps).cast("int"),
        )
        df = df.withColumn(
            "amount_close_to_delta_dest",
            (F.abs(F.col("amount") - (F.col("newbalanceDest") - F.col("oldbalanceDest"))) <= eps).cast("int"),
        )

        # Ratios / logs
        df = df.withColumn("log_amount", F.log1p(F.col("amount")))
        df = df.withColumn("amt_over_old_org", F.col("amount") / (F.col("oldbalanceOrg") + F.lit(1.0)))
        df = df.withColumn("amt_over_old_dest", F.col("amount") / (F.col("oldbalanceDest") + F.lit(1.0)))

        # for all types 
        for t in TYPES:
            col = f"type_{t}"
            df = df.withColumn(col, (F.col("type") == F.lit(t)).cast("int"))

        # Velocity features
        # Windows keyed by origin and destination names, ordered by time
        w_orig_1d   = Window.partitionBy("nameOrig").orderBy(F.col("step")).rangeBetween(-24, 0)
        w_orig_7d   = Window.partitionBy("nameOrig").orderBy(F.col("step")).rangeBetween(-24*7, 0)
        w_orig_30d  = Window.partitionBy("nameOrig").orderBy(F.col("step")).rangeBetween(-24*30, 0)

        w_dest_1d   = Window.partitionBy("nameDest").orderBy(F.col("step")).rangeBetween(-24, 0)
        w_dest_7d   = Window.partitionBy("nameDest").orderBy(F.col("step")).rangeBetween(-24*7, 0)
        w_dest_30d  = Window.partitionBy("nameDest").orderBy(F.col("step")).rangeBetween(-24*30, 0)

        df = df.withColumn("tx_count_orig_1d",  F.count(F.lit(1)).over(w_orig_1d))
        df = df.withColumn("tx_count_orig_7d",  F.count(F.lit(1)).over(w_orig_7d))
        df = df.withColumn("sum_amount_orig_7d", F.sum(F.col("amount")).over(w_orig_7d))
        df = df.withColumn("max_amount_orig_30d", F.max(F.col("amount")).over(w_orig_30d))

        df = df.withColumn("tx_count_dest_1d",  F.count(F.lit(1)).over(w_dest_1d))
        df = df.withColumn("tx_count_dest_7d",  F.count(F.lit(1)).over(w_dest_7d))
        df = df.withColumn("sum_amount_dest_7d", F.sum(F.col("amount")).over(w_dest_7d))
        df = df.withColumn("max_amount_dest_30d", F.max(F.col("amount")).over(w_dest_30d))

        # Final columns 
        base_cols = [
            "step", "amount",
            "oldbalanceOrg", "newbalanceOrig", "delta_bal",
            "oldbalanceDest", "newbalanceDest", "delta_bal_dest",
            "is_weekend",
            "dest_zero_before", "dest_zero_after",
            "amount_close_to_delta_orig", "amount_close_to_delta_dest",
            "log_amount", "amt_over_old_org", "amt_over_old_dest",
        ]
        type_cols = [f"type_{t}" for t in TYPES]
        vel_cols = [
            "tx_count_orig_1d", "tx_count_orig_7d", "sum_amount_orig_7d", "max_amount_orig_30d",
            "tx_count_dest_1d", "tx_count_dest_7d", "sum_amount_dest_7d", "max_amount_dest_30d",
        ]

        selected = base_cols + type_cols + vel_cols + ["isFraud"]
        return df.select(*selected)
