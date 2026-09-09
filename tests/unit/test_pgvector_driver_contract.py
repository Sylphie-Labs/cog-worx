"""The three pgvector facts ``PgLatentStore`` is written against, pinned without a database.

Every one of them changed in pgvector 0.5.0 with no deprecation, and the previous uncapped
``pgvector>=0.3.6`` let that release in silently: the adapter became unimportable, and its write
and read paths both raised.  Nothing caught it, because the only tests that exercise
``pg_latent`` are marked ``integration`` and the integration job runs on ``main`` alone
(``.github/workflows/ci.yml``: ``if: github.ref == 'refs/heads/main'``).

These assertions need no Postgres and no numpy, so they run in the deterministic tier on every
PR.  If the pin in ``pyproject.toml`` is lifted to a release that moves this surface again, this
file goes red on the PR that lifts it rather than on ``main`` weeks later.
"""

from __future__ import annotations

import pytest


def test_vector_is_exported_from_the_top_level_package() -> None:
    """0.5.0 moved ``Vector`` out of ``pgvector.psycopg``; the adapter imports it from the root."""
    from pgvector import Vector

    assert Vector is not None


def test_psycopg_submodule_still_exports_the_async_registrar() -> None:
    """``register_vector_async`` stayed in ``pgvector.psycopg`` across the move."""
    from pgvector.psycopg import register_vector_async

    assert callable(register_vector_async)


def test_vector_accepts_a_list_and_rejects_a_tuple() -> None:
    """Why ``put`` wraps in ``list(...)``.

    ``LatentRecord.embedding`` is a ``tuple[float, ...]``.  0.4 accepted it; 0.5 accepts only
    ``list``/``ndarray`` and raises ``ValueError`` otherwise.
    """
    from pgvector import Vector

    assert Vector([1.0, 2.0, 3.0]) is not None

    with pytest.raises(ValueError, match="expected list or ndarray"):
        # Deliberately the wrong type: passing a tuple is the thing being asserted about, and
        # mypy sees pgvector's real signature here (the typecheck session installs the `sizing`
        # extra, so numpy resolves). Under --strict this ignore is also a second tripwire — if a
        # future pgvector accepts tuples again, `warn_unused_ignores` fails on this line.
        Vector((1.0, 2.0, 3.0))  # type: ignore[arg-type]


def test_a_loaded_vector_is_not_iterable_and_unpacks_via_to_list() -> None:
    """Why ``search`` calls ``.to_list()``.

    The psycopg loader returns whatever ``Vector.from_text`` produces.  Through 0.4 that was an
    ndarray, which the old ``tuple(float(v) for v in row[1])`` could iterate; from 0.5 it is a
    ``Vector``, which cannot be iterated at all.
    """
    from pgvector import Vector

    loaded = Vector.from_text("[1,2,3]")

    assert not hasattr(loaded, "__iter__"), (
        "a loaded Vector became iterable again -- re-check whether pg_latent.search should still "
        "call .to_list()"
    )
    assert loaded.to_list() == [1.0, 2.0, 3.0]


def test_the_loader_the_adapter_relies_on_returns_that_type() -> None:
    """Ties the two facts above to the class psycopg actually calls on a read."""
    from pgvector import Vector
    from pgvector.psycopg.vector import VectorLoader

    assert VectorLoader.load.__annotations__.get("return") in {"Vector | None", Vector | None}
