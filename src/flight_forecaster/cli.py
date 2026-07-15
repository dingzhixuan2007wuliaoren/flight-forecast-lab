from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import uvicorn

from flight_forecaster.api import get_service
from flight_forecaster.data import (
    generate_demo_ontime_data,
    generate_demo_price_data,
    load_ontime_csv,
    load_price_csv,
)
from flight_forecaster.schemas import OnTimeRequest, PriceRequest
from flight_forecaster.service import PredictionService
from flight_forecaster.training import train_models


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _train_demo(args: argparse.Namespace) -> None:
    price = generate_demo_price_data(rows=args.price_rows, seed=args.seed)
    on_time = generate_demo_ontime_data(rows=args.ontime_rows, seed=args.seed + 1)
    if args.save_demo_data:
        data_dir = Path(args.save_demo_data)
        data_dir.mkdir(parents=True, exist_ok=True)
        price.to_csv(data_dir / "demo_price.csv", index=False)
        on_time.to_csv(data_dir / "demo_ontime.csv", index=False)
    bundle = train_models(
        price,
        on_time,
        args.output,
        data_mode="synthetic_demo",
        random_state=args.seed,
    )
    print(json.dumps(bundle["metrics"], indent=2, sort_keys=True))
    print(f"Saved model and report to {Path(args.output).resolve()}")


def _train_csv(args: argparse.Namespace) -> None:
    bundle = train_models(
        load_price_csv(args.price_csv),
        load_ontime_csv(args.ontime_csv),
        args.output,
        data_mode="user_csv",
        random_state=args.seed,
    )
    print(json.dumps(bundle["metrics"], indent=2, sort_keys=True))


def _predict_price(args: argparse.Namespace) -> None:
    service = PredictionService(args.model_dir)
    result = service.predict_price(PriceRequest.model_validate(_json(args.input)))
    print(result.model_dump_json(indent=2))


def _predict_ontime(args: argparse.Namespace) -> None:
    service = PredictionService(args.model_dir)
    result = service.predict_ontime(OnTimeRequest.model_validate(_json(args.input)))
    print(result.model_dump_json(indent=2))


def _serve(args: argparse.Namespace) -> None:
    os.environ["MODEL_DIR"] = str(Path(args.model_dir).resolve())
    get_service.cache_clear()
    uvicorn.run("flight_forecaster.api:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flight-forecast", description="Train and serve Flight Forecast Lab models"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "train-demo",
        help="train the fare model and both on-time variants on reproducible demo data",
    )
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--price-rows", type=int, default=6_000)
    demo.add_argument("--ontime-rows", type=int, default=8_000)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--save-demo-data")
    demo.set_defaults(handler=_train_demo)

    csv_train = subparsers.add_parser("train-csv", help="train from two contract-compatible CSVs")
    csv_train.add_argument("--price-csv", required=True)
    csv_train.add_argument("--ontime-csv", required=True)
    csv_train.add_argument("--output", default="artifacts/custom")
    csv_train.add_argument("--seed", type=int, default=42)
    csv_train.set_defaults(handler=_train_csv)

    price = subparsers.add_parser("predict-price", help="predict a fare from a JSON request")
    price.add_argument("--input", required=True)
    price.add_argument("--model-dir", default="artifacts/demo")
    price.set_defaults(handler=_predict_price)

    ontime = subparsers.add_parser("predict-on-time", help="predict on-time probability from JSON")
    ontime.add_argument("--input", required=True)
    ontime.add_argument("--model-dir", default="artifacts/demo")
    ontime.set_defaults(handler=_predict_ontime)

    serve = subparsers.add_parser("serve", help="start the API and dashboard")
    serve.add_argument("--model-dir", default="artifacts/demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
