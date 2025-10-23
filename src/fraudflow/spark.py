from __future__ import annotations
import logging
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

class SparkSessionManager:
    def __init__(self, app_name: str = "FraudFlow") -> None:
        self.app_name = app_name
        self._spark: SparkSession | None = None

    @property
    def spark(self) -> SparkSession:
        if self._spark is None:
            logger.info("Creating local SparkSession--")
            self._spark = (
                SparkSession.builder
                .appName(self.app_name)
                .master("local[*]")
                .config("spark.driver.memory", "6g") 
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.driver.memory", "3g")
                .config("spark.local.ip", "127.0.0.1")
                .getOrCreate()
            )
        return self._spark

    def stop(self) -> None:
        if self._spark is not None:
            logger.info("Stopping SparkSession…")
            self._spark.stop()
            self._spark = None
