from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

__all__ = ["RecallQuery"]


class RecallQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str | None = None  # BM25 channel query string
    embedding: tuple[float, ...] | None = None  # dense channel query vector
    anchor_entities: tuple[str, ...] = ()  # graph channel entity IDs to expand from
    session_id: str | None = None  # temporal channel: session to retrieve episodes for
    scope: str | None = None  # claim scope filter (Pod 2.4); None = all scopes
    as_of: datetime | None = None  # bi-temporal filter; None = currently-valid claims
    k: int = 20  # per-channel fetch depth before fusion
