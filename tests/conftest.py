from pathlib import Path

import pytest

from flight_forecaster.data import generate_demo_ontime_data, generate_demo_price_data
from flight_forecaster.training import train_models


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("models")
    train_models(
        generate_demo_price_data(rows=1_500, seed=42),
        generate_demo_ontime_data(rows=2_000, seed=43),
        output,
        data_mode="pytest_synthetic",
        random_state=42,
    )
    return output
