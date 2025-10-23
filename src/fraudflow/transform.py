from __future__ import annotations
import logging
from pyspark.sql import DataFrame, functions as F

logger = logging.getLogger(__name__)

class PaymentTransformer:
    def to_silver(self, bronze_df: DataFrame) -> DataFrame:
        logger.info("Transforming to Silver (type casts, sanity flags, subscription framing)…")
        df = bronze_df
        df = df.withColumn("is_renewal", (F.col("type") == F.lit("PAYMENT")).cast("int"))
        df = df.withColumn("delta_bal", F.col("newbalanceOrig") - F.col("oldbalanceOrg"))
        df = df.withColumn("is_weekend", (F.col("step") % 7 >= 5).cast("int"))
        return df