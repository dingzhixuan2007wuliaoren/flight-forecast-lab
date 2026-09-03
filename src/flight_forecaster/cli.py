from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn

from flight_forecaster.api import get_service
from flight_forecaster.data import (
    generate_demo_ontime_data,
    generate_demo_price_data,
    load_ontime_csv,
    load_price_csv,
)
from flight_forecaster.hotel_model import (
    generate_demo_hotel_price_data,
    load_hotel_price_model,
    predict_hotel_price,
    save_hotel_price_model,
    train_hotel_price_model,
)
from flight_forecaster.schemas import OnTimeRequest, PriceRequest
from flight_forecaster.service import PredictionService
from flight_forecaster.training import train_models


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _train_demo(args: argparse.Namespace) -> None:
    price = generate_demo_price_data(rows=args.price_rows, seed=args.seed)
    on_time = generate_demo_ontime_data(rows=args.ontime_rows, seed=args.seed + 1)
    hotel_price = generate_demo_hotel_price_data(rows=args.hotel_rows, seed=args.seed + 2)
    if args.save_demo_data:
        data_dir = Path(args.save_demo_data)
        data_dir.mkdir(parents=True, exist_ok=True)
        price.to_csv(data_dir / "demo_price.csv", index=False)
        on_time.to_csv(data_dir / "demo_ontime.csv", index=False)
        hotel_price.to_csv(data_dir / "demo_hotel_price.csv", index=False)
    bundle = train_models(
        price,
        on_time,
        args.output,
        data_mode="synthetic_demo",
        random_state=args.seed,
    )
    hotel_bundle = train_hotel_price_model(
        hotel_price,
        data_mode="synthetic_demo",
        random_state=args.seed + 2,
    )
    save_hotel_price_model(hotel_bundle, args.output)
    print(
        json.dumps(
            {
                "flight": bundle["metrics"],
                "hotel_price": hotel_bundle["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Saved model and report to {Path(args.output).resolve()}")


def _train_csv(args: argparse.Namespace) -> None:
    bundle = train_models(
        load_price_csv(args.price_csv),
        load_ontime_csv(args.ontime_csv),
        args.output,
        data_mode="user_csv",
        random_state=args.seed,
    )
    metrics: dict[str, Any] = {"flight": bundle["metrics"]}
    if args.hotel_price_csv:
        hotel_bundle = train_hotel_price_model(
            pd.read_csv(args.hotel_price_csv),
            data_mode="user_csv",
            random_state=args.seed,
        )
        save_hotel_price_model(hotel_bundle, args.output)
        metrics["hotel_price"] = hotel_bundle["metrics"]
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _predict_price(args: argparse.Namespace) -> None:
    service = PredictionService(args.model_dir)
    result = service.predict_price(PriceRequest.model_validate(_json(args.input)))
    print(result.model_dump_json(indent=2))


def _predict_ontime(args: argparse.Namespace) -> None:
    service = PredictionService(args.model_dir)
    result = service.predict_ontime(OnTimeRequest.model_validate(_json(args.input)))
    print(result.model_dump_json(indent=2))


def _predict_hotel_price(args: argparse.Namespace) -> None:
    payload = _json(args.input)
    if "quote_time" in payload:
        raise ValueError("quote_time is supplied by the application, not the input file")
    result = predict_hotel_price(
        load_hotel_price_model(args.model_dir),
        payload,
        quote_time=datetime.now(UTC),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True, ensure_ascii=False))


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
        help=(
            "train the fare, hotel-price, and both on-time variants on "
            "reproducible demo data"
        ),
    )
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--price-rows", type=int, default=6_000)
    demo.add_argument("--ontime-rows", type=int, default=8_000)
    demo.add_argument("--hotel-rows", type=int, default=6_000)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--save-demo-data")
    demo.set_defaults(handler=_train_demo)

    csv_train = subparsers.add_parser(
        "train-csv",
        help="train from contract-compatible fare/on-time CSVs and optional hotel CSV",
    )
    csv_train.add_argument("--price-csv", required=True)
    csv_train.add_argument("--ontime-csv", required=True)
    csv_train.add_argument(
        "--hotel-price-csv",
        help="optional contract-compatible hotel price observations",
    )
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

    hotel_price = subparsers.add_parser(
        "predict-hotel-price",
        help="estimate a hotel stay from the independent local model",
    )
    hotel_price.add_argument("--input", required=True)
    hotel_price.add_argument("--model-dir", default="artifacts/demo")
    hotel_price.set_defaults(handler=_predict_hotel_price)

    serve = subparsers.add_parser("serve", help="start the API and dashboard")
    serve.add_argument("--model-dir", default="artifacts/demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
