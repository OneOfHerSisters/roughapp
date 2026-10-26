"""Seed lookup tables for strategies and tie-breaking policies."""

from __future__ import annotations

from sqlalchemy.orm import Session

from roughapp.db.models import McdTiePolicy, QrTiePolicy, Strategy, TreeAlgorithm

STRATEGY_ROWS = [
    ("mcd", "Most common decision (majority vote per conflict class)"),
    ("rough-approximation", "Qualitative removal: keep the decision whose class has the highest accuracy of approximation"),
    ("gd", "Generalized decision (integer codes for decision sets)"),
]

MCD_TIE_POLICY_ROWS = [
    ("min", "Pick the smallest decision value on ties"),
    ("keep", "Leave tied conflict classes unresolved"),
    ("first", "Pick the first candidate in index order"),
    ("random", "Draw uniformly at random among tied decisions"),
    ("raise", "Abort with an error on ties"),
]

QR_TIE_POLICY_ROWS = [
    ("keep", "Keep all tied objects in the conflict class"),
    ("random", "Remove a random subset of tied objects"),
    ("raise", "Abort with an error on ties"),
]

TREE_ALGORITHM_ROWS = [
    ("id3", "ID3 — information-gain tree with one branch per attribute value (Quinlan)"),
]


def _upsert_lookup(db: Session, model, rows: list[tuple[str, str]]) -> None:
    for name, description in rows:
        existing = db.query(model).filter_by(name=name).first()
        if existing:
            existing.description = description
        else:
            db.add(model(name=name, description=description))


def seed_reference_tables(db: Session) -> None:
    _upsert_lookup(db, Strategy, STRATEGY_ROWS)
    _upsert_lookup(db, McdTiePolicy, MCD_TIE_POLICY_ROWS)
    _upsert_lookup(db, QrTiePolicy, QR_TIE_POLICY_ROWS)
    _upsert_lookup(db, TreeAlgorithm, TREE_ALGORITHM_ROWS)
    db.commit()
