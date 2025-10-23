from __future__ import annotations
import logging
from pathlib import Path
from pyspark.sql import DataFrame
from .spark import SparkSessionManager
from .schemas import payments_schema

logger = logging.getLogger(__name__)

class DataIngestor:
    def __init__(self, spark_mgr: SparkSessionManager) -> None:
        self.spark = spark_mgr.spark

    def read_csv(self, path: Path) -> DataFrame:
        logger.info(f"Reading CSV: {path}")
        return (
            self.spark.read
            .option("header", True)
            .schema(payments_schema)
            .csv(str(path))
        )

    def write_parquet(self, df: DataFrame, out_dir: Path, mode: str = "overwrite") -> None:
        logger.info(f"Writing Parquet to: {out_dir}")
        (
            df.write
            .mode(mode)
            .parquet(str(out_dir))
        )