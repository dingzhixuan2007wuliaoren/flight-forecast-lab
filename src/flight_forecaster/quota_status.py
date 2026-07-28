"""Credential-free quota snapshots read only from durable local ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

QuotaDataBasis = Literal[
    "local_hard_limit",
    "provider_snapshot",
    "conservative_minimum",
]


@dataclass(frozen=True, slots=True)
class QuotaLedgerSnapshot:
    available: bool
    used: int | None = None
    limit: int | None = None
    remaining: int | None = None
    period_key: str | None = None
    data_basis: QuotaDataBasis | None = None
    observed_at: datetime | None = None
    reset_at: datetime | None = None

    @classmethod
    def unavailable(cls) -> QuotaLedgerSnapshot:
        return cls(available=False)
