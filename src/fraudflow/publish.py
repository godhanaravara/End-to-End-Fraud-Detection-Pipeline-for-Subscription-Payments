from __future__ import annotations
from pathlib import Path
import duckdb

def _sql_quote(path: str) -> str:
    return path.replace("'", "''")

def publish(
    gold: str = "data/gold/scores.parquet",
    db: str = "reports/daily_summary.sqlite",
    threshold: float | None = None,
) -> None:
    Path(db).parent.mkdir(parents=True, exist_ok=True)

    gold_path = Path(gold)
    gold_glob = gold if gold_path.is_file() else f"{gold.rstrip('/')}/**/*.parquet"

    con = duckdb.connect()
    try:
        # Attach the SQLite file
        con.execute(f"ATTACH DATABASE '{_sql_quote(db)}' AS sqlite (TYPE SQLITE)")

        # Stream Parquet
        con.execute(
            f"CREATE OR REPLACE TABLE sqlite.scores AS "
            f"SELECT * FROM read_parquet('{_sql_quote(gold_glob)}')"
        )

        if threshold is None:
            th = con.execute(
                "SELECT percentile_cont(0.99) WITHIN GROUP (ORDER BY score) FROM sqlite.scores"
            ).fetchone()[0]
        else:
            th = float(threshold)

        # Summary table in SQLite
        con.execute(
            f"""
            CREATE OR REPLACE TABLE sqlite.summary AS
            SELECT
              COUNT(*)                                   AS n,
              AVG(score)                                 AS avg_score,
              SUM(COALESCE(isFraud, 0))                  AS fraud_count,
              SUM(CASE WHEN score >= {th} THEN 1 ELSE 0 END) AS alerts
            FROM sqlite.scores
            """
        )
    finally:
        con.execute("DETACH DATABASE sqlite")
        con.close()

