"""API test fixtures.

The app is created against a temporary SQLite database so tests never touch
data/roughapp.db.  DATABASE_URL must be set before roughapp.db.engine is
imported (the engine is built at import time), which is why the environment
is prepared at module import instead of inside a fixture.
"""
import os
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="roughapp_api_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"

import pytest
from fastapi.testclient import TestClient

from roughapp.config import get_settings

get_settings.cache_clear()

from roughapp.api import app  # noqa: E402  (needs the env var above)


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:  # context manager runs the startup event
        yield c


@pytest.fixture()
def csv_bytes() -> bytes:
    # Table 1 from the thesis: one conflict class {x3, x9}, γ = 7/9.
    return (
        "a,b,c,d,dec\n"
        "0,0,0,0,0\n"
        "0,0,1,1,0\n"
        "1,1,1,0,0\n"
        "1,1,2,0,0\n"
        "0,1,0,1,1\n"
        "1,1,0,2,1\n"
        "0,1,1,2,2\n"
        "1,0,1,0,2\n"
        "1,1,1,0,2\n"
    ).encode()


def register_and_login(client: TestClient, email: str) -> dict:
    r = client.post("/api/auth/register", json={"email": email, "password": "password1"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def upload(client: TestClient, csv: bytes, headers=None, decision_col="dec", name="t1.csv"):
    return client.post(
        "/api/datasets",
        files={"file": (name, csv, "text/csv")},
        data={"decision_col": decision_col},
        headers=headers or {},
    )
