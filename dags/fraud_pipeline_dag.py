from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJ="/home/godhanaravara/programming/subscription-payment-fraud-detection/fraudflow-local"
PY=f"{PROJ}/.venv-airflow/bin/python"
CSV=f"{PROJ}/data/raw/PS_20174392719_1491204439457_log.csv"

with DAG(
    dag_id="fraudflow_local",
    start_date=datetime(2025,10,21),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    extract_transform_train = BashOperator(
        task_id="extract_transform_train",
        bash_command=f"cd {PROJ} && {PY} -m fraudflow.cli run-all --csv-path {CSV}",
        env={"SPARK_LOCAL_IP":"127.0.0.1","PYSPARK_PYTHON":PY},
    )
    score = BashOperator(
        task_id="score",
        bash_command=f"cd {PROJ} && {PY} -m fraudflow.cli score",
        env={"SPARK_LOCAL_IP":"127.0.0.1","PYSPARK_PYTHON":PY},
    )
    publish = BashOperator(
        task_id="publish",
        bash_command=(
            f"cd {PROJ} && {PY} -m fraudflow.cli publish-sqlite "
            f"--gold {PROJ}/data/gold/scores.parquet "
            f"--db {PROJ}/reports/daily_summary.sqlite"
        ),
    )

    extract_transform_train >> score >> publish
