import typer
from .logging_config import configure_logging

app = typer.Typer()

@app.command()
def run_all(csv_path: str | None = None):
    configure_logging()
    from .pipeline import FraudPipeline # lazy import load settings when needed
    pipe = FraudPipeline()
    bronze = pipe.extract(csv_path)
    silver = pipe.transform(bronze)
    feats = pipe.build_features(silver)
    metrics = pipe.train(feats)
    typer.echo({"train": metrics})

@app.command()
def score():
    configure_logging()
    from .pipeline import FraudPipeline # lazy import load settings when needed
    pipe = FraudPipeline()
    feats = pipe.build_features(None)
    pipe.score(feats)

@app.command()
def publish_sqlite(gold: str = "data/gold/scores.parquet", db: str = "reports/daily_summary.sqlite"):
    from .publish import publish
    publish(gold=gold, db=db)
    print(f"Wrote {db}")

if __name__ == "__main__":
    app()