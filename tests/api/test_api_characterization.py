"""Characterization tests: pin down the public API behavior before the
layered refactoring.  Every assertion here reflects behavior observed on the
pre-refactoring code, not aspirations."""
import pytest

from tests.api.conftest import register_and_login, upload


def _provenance(d: dict) -> dict:
    return d.get("provenance") or {}


def _prov_kind(d: dict) -> str | None:
    return _provenance(d).get("kind")


def _source_dataset_id(d: dict) -> int | None:
    return _provenance(d).get("source_dataset_id")


class TestAuth:
    def test_register_login_me(self, client):
        h = register_and_login(client, "char_auth@t.com")
        r = client.post("/api/auth/login", json={"email": "char_auth@t.com", "password": "password1"})
        assert r.status_code == 200
        me = client.get("/api/auth/me", headers=h)
        assert me.status_code == 200
        assert me.json()["email"] == "char_auth@t.com"

    def test_short_password_422(self, client):
        r = client.post("/api/auth/register", json={"email": "short@t.com", "password": "short"})
        assert r.status_code == 422

    def test_duplicate_email_409(self, client):
        register_and_login(client, "dup@t.com")
        r = client.post("/api/auth/register", json={"email": "dup@t.com", "password": "password1"})
        assert r.status_code == 409

    def test_wrong_password_401(self, client):
        register_and_login(client, "wrongpw@t.com")
        r = client.post("/api/auth/login", json={"email": "wrongpw@t.com", "password": "password2"})
        assert r.status_code == 401


class TestDatasets:
    def test_builtin_datasets_public(self, client):
        r = client.get("/api/datasets")
        assert r.status_code == 200
        assert len(r.json()) > 0  # seeded builtins

    def test_upload_requires_decision_col_in_csv(self, client, csv_bytes):
        h = register_and_login(client, "up1@t.com")
        r = upload(client, csv_bytes, h, decision_col="nope")
        assert r.status_code == 422

    def test_upload_and_metadata(self, client, csv_bytes):
        h = register_and_login(client, "up2@t.com")
        r = upload(client, csv_bytes, h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rows"] == 9
        assert body["cols"] == 5
        assert body["decision_col"] == "dec"
        assert body["gamma_actual"] == pytest.approx(7 / 9, abs=1e-3)

    def test_private_dataset_hidden_from_others(self, client, csv_bytes):
        h1 = register_and_login(client, "own1@t.com")
        h2 = register_and_login(client, "own2@t.com")
        ds_id = upload(client, csv_bytes, h1).json()["id"]
        assert client.get(f"/api/datasets/{ds_id}", headers=h1).status_code == 200
        assert client.get(f"/api/datasets/{ds_id}", headers=h2).status_code == 404
        assert ds_id not in [d["id"] for d in client.get("/api/datasets", headers=h2).json()]

    def test_preview(self, client, csv_bytes):
        h = register_and_login(client, "prev@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        r = client.get(f"/api/datasets/{ds_id}/preview?offset=0&limit=5", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["columns"] == ["a", "b", "c", "d", "dec"]
        assert body["total"] == 9
        assert len(body["rows"]) == 5

    def test_analyze(self, client, csv_bytes):
        h = register_and_login(client, "ana@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        r = client.get(f"/api/datasets/{ds_id}/analyze", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["inconsistent_class_count"] == 1
        assert body["quality"] == pytest.approx(7 / 9, abs=1e-3)

    def test_attribute_importance(self, client, csv_bytes):
        h = register_and_login(client, "imp@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        r = client.get(f"/api/datasets/{ds_id}/attribute-importance", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["gamma_full"] == pytest.approx(7 / 9, abs=1e-3)
        assert len(body["attributes"]) == 4

    def test_dataset_tree_preview_and_save(self, client, csv_bytes):
        """Preview does not persist; save stores metrics and artifact."""
        h = register_and_login(client, "dstree@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        r = client.get(f"/api/datasets/{ds_id}/tree", headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["stats"]["depth"] >= 1
        assert "text" in body
        assert "-->" in body["text"]
        ds = client.get(f"/api/datasets/{ds_id}", headers=h).json()
        assert ds["depth_actual"] is None

        saved = client.post(f"/api/datasets/{ds_id}/tree/build", headers=h)
        assert saved.status_code == 200, saved.text
        assert saved.json()["depth"] == body["stats"]["depth"]
        assert "text" in saved.json()
        text = client.get(f"/api/trees/{saved.json()['tree_run_id']}/text", headers=h)
        assert text.status_code == 200
        assert text.json()["text"] == saved.json()["text"]
        ds = client.get(f"/api/datasets/{ds_id}", headers=h).json()
        assert ds["depth_actual"] == body["stats"]["depth"]

    def test_dataset_tree_public_for_builtin(self, client):
        builtin_id = client.get("/api/datasets").json()[0]["id"]
        fresh = type(client)(client.app)
        with fresh:
            r = fresh.get(f"/api/datasets/{builtin_id}/tree")
            assert r.status_code == 200

    def test_builtin_delete_forbidden(self, client):
        builtin_id = client.get("/api/datasets").json()[0]["id"]
        h = register_and_login(client, "deleter@t.com")
        r = client.delete(f"/api/datasets/{builtin_id}", headers=h)
        assert r.status_code == 403

    def test_delete_own(self, client, csv_bytes):
        h = register_and_login(client, "delown@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        assert client.delete(f"/api/datasets/{ds_id}", headers=h).status_code == 204
        assert client.get(f"/api/datasets/{ds_id}", headers=h).status_code == 404

    def test_introduce_rejects_target_at_or_above_gamma(self, client, csv_bytes):
        h = register_and_login(client, "introfail@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        r = client.post(
            f"/api/datasets/{ds_id}/introduce",
            json={"method": "until_quality", "target_gamma": 0.9},
            headers=h,
        )
        assert r.status_code == 422

    def test_introduce_until_quality(self, client, csv_bytes):
        h = register_and_login(client, "intro@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        r = client.post(
            f"/api/datasets/{ds_id}/introduce",
            json={"method": "until_quality", "target_gamma": 0.5, "random_state": 42},
            headers=h,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert _source_dataset_id(body) == ds_id
        assert body["gamma_actual"] <= 0.5

    def test_duplicate_detection_and_removal(self, client):
        h = register_and_login(client, "dedup@t.com")
        # 6 rows: (1,1,P)x2, (2,2,N)x3, (3,3,P) → 3 exact duplicates
        csv = b"a,b,dec\n1,1,P\n1,1,P\n2,2,N\n2,2,N\n2,2,N\n3,3,P\n"
        ds_id = upload(client, csv, h, name="dup.csv").json()["id"]
        analysis = client.get(f"/api/datasets/{ds_id}/analyze", headers=h).json()
        assert analysis["duplicate_rows"] == 3

        deduped = client.post(f"/api/datasets/{ds_id}/deduplicate", headers=h)
        assert deduped.status_code == 200, deduped.text
        new_id = deduped.json()["id"]
        assert deduped.json()["rows"] == 3
        assert _source_dataset_id(deduped.json()) == ds_id
        # the deduped table has no duplicates left
        assert client.get(f"/api/datasets/{new_id}/analyze", headers=h).json()["duplicate_rows"] == 0
        # removing again is rejected
        assert client.post(f"/api/datasets/{new_id}/deduplicate", headers=h).status_code == 422

    def test_complete_without_missing_422(self, client, csv_bytes):
        h = register_and_login(client, "compl@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        r = client.post(f"/api/datasets/{ds_id}/complete", json={"method": "fill"}, headers=h)
        assert r.status_code == 422

    def test_complete_fill(self, client):
        h = register_and_login(client, "compl2@t.com")
        csv = b"a,b,dec\n1,?,P\n1,2,P\n2,2,N\n2,2,N\n"
        ds_id = upload(client, csv, h, name="miss.csv").json()["id"]
        summary = client.get(f"/api/datasets/{ds_id}/missing-summary", headers=h).json()
        assert summary["total_missing"] == 1
        r = client.post(f"/api/datasets/{ds_id}/complete", json={"method": "fill"}, headers=h)
        assert r.status_code == 200, r.text
        assert _source_dataset_id(r.json()) == ds_id


    def test_delete_derived_dataset(self, client, csv_bytes):
        h = register_and_login(client, "delderived@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        derived = client.post(
            f"/api/datasets/{ds_id}/introduce",
            json={"method": "until_quality", "target_gamma": 0.5, "random_state": 42},
            headers=h,
        ).json()
        assert _source_dataset_id(derived) == ds_id
        assert client.delete(f"/api/datasets/{derived['id']}", headers=h).status_code == 204
        # Parent stays; only the derived copy is removed.
        assert client.get(f"/api/datasets/{ds_id}", headers=h).status_code == 200

    def test_delete_derived_from_builtin(self, client):
        h = register_and_login(client, "delbuiltinchild@t.com")
        cars = next(d for d in client.get("/api/datasets", headers=h).json() if d["name"] == "cars")
        derived = client.post(
            f"/api/datasets/{cars['id']}/introduce",
            json={"method": "until_quality", "target_gamma": 0.5, "random_state": 42},
            headers=h,
        ).json()
        assert _source_dataset_id(derived) == cars["id"]
        assert client.delete(f"/api/datasets/{derived['id']}", headers=h).status_code == 204


class TestGuestSessions:
    def test_guest_upload_and_adoption(self, client, csv_bytes):
        # a cookie-holding guest uploads a dataset...
        guest = type(client)(client.app)
        with guest:
            r = upload(guest, csv_bytes)
            assert r.status_code == 200, r.text
            ds_id = r.json()["id"]
            assert "roughapp_guest" in r.cookies or guest.cookies.get("roughapp_guest")
            assert ds_id in [d["id"] for d in guest.get("/api/datasets").json()]
            # ...then registers: the dataset moves to the account
            reg = guest.post("/api/auth/register", json={"email": "adopted@t.com", "password": "password1"})
            assert reg.status_code == 201
            h = {"Authorization": f"Bearer {reg.json()['access_token']}"}
            assert ds_id in [d["id"] for d in guest.get("/api/datasets", headers=h).json()]

    def test_anonymous_without_cookie_sees_only_builtins(self, client, csv_bytes):
        fresh = type(client)(client.app)
        with fresh:
            for d in fresh.get("/api/datasets").json():
                assert _prov_kind(d) in (None, "upload")
                assert _source_dataset_id(d) is None


class TestExperiments:
    @pytest.fixture()
    def ds_and_headers(self, client, csv_bytes):
        h = register_and_login(client, f"exp{id(self)}@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        return ds_id, h

    def test_create_all_strategies(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        r = client.post("/api/experiments", json={"dataset_id": ds_id, "random_state": 42}, headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["gamma_before"] == pytest.approx(7 / 9, abs=1e-3)
        assert body["rows_before"] == 9
        strategies = {x["strategy"]: x for x in body["results"]}
        assert set(strategies) == {"mcd", "rough-approximation", "gd"}
        assert strategies["mcd"]["rows_removed"] == 9 - strategies["mcd"]["rows_after"]
        assert strategies["gd"]["gamma_after"] == 1.0

    def test_unknown_strategy_422(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        r = client.post("/api/experiments", json={"dataset_id": ds_id, "strategies": ["greedy"]}, headers=h)
        assert r.status_code == 422

    def test_unknown_tie_policy_422(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        r = client.post("/api/experiments", json={"dataset_id": ds_id, "tie_policy": "wat"}, headers=h)
        assert r.status_code == 422

    def test_get_list_and_isolation(self, client, ds_and_headers, csv_bytes):
        ds_id, h = ds_and_headers
        exp_id = client.post("/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h).json()["id"]
        assert client.get(f"/api/experiments/{exp_id}", headers=h).status_code == 200
        assert exp_id in [e["id"] for e in client.get("/api/experiments", headers=h).json()]
        stranger = register_and_login(client, "stranger@t.com")
        assert client.get(f"/api/experiments/{exp_id}", headers=stranger).status_code == 404

    def test_builtin_experiment_in_history_and_isolated(self, client):
        h = register_and_login(client, f"builtin{id(self)}@t.com")
        cars = next(d for d in client.get("/api/datasets", headers=h).json() if d["name"] == "cars")
        exp_id = client.post(
            "/api/experiments",
            json={"dataset_id": cars["id"], "strategies": ["mcd"]},
            headers=h,
        ).json()["id"]
        assert exp_id in [e["id"] for e in client.get("/api/experiments", headers=h).json()]
        stranger = register_and_login(client, f"builtin-stranger{id(self)}@t.com")
        assert client.get(f"/api/experiments/{exp_id}", headers=stranger).status_code == 404

    def test_dataset_delete_manifest_and_cascade(self, client, ds_and_headers):
        """Deleting an experiment's input dataset: 409 with a manifest first,
        then cascade=true deletes the run records but keeps their result
        tables (they are independent datasets)."""
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h,
        ).json()["id"]

        r = client.delete(f"/api/datasets/{ds_id}", headers=h)
        assert r.status_code == 409
        body = r.json()
        assert body["cascade_required"] is True
        assert [e["id"] for e in body["manifest"]["experiments"]] == [exp_id]
        assert body["manifest"]["experiments"][0]["strategies"] == ["mcd"]

        outputs_before = [
            d["id"] for d in client.get("/api/datasets", headers=h).json()
            if _prov_kind(d) == "experiment_output"
        ]
        r = client.delete(f"/api/datasets/{ds_id}?cascade=true", headers=h)
        assert r.status_code == 204
        assert client.get(f"/api/experiments/{exp_id}", headers=h).status_code == 404
        remaining = [d["id"] for d in client.get("/api/datasets", headers=h).json()]
        assert ds_id not in remaining
        for out_id in outputs_before:
            assert out_id in remaining  # result tables survive the cascade

    def test_dataset_delete_keeps_derived_children(self, client, csv_bytes):
        """A dataset can be deleted even when derived tables exist; they
        survive with the provenance link severed."""
        h = register_and_login(client, "derivedsurvive@t.com")
        ds_id = upload(client, csv_bytes, h).json()["id"]
        child = client.post(
            f"/api/datasets/{ds_id}/introduce",
            json={"method": "until_quality", "target_gamma": 0.5, "random_state": 42},
            headers=h,
        ).json()
        assert _source_dataset_id(child) == ds_id

        assert client.delete(f"/api/datasets/{ds_id}", headers=h).status_code == 204
        survivor = client.get(f"/api/datasets/{child['id']}", headers=h)
        assert survivor.status_code == 200
        assert _prov_kind(survivor.json()) == "derived"
        assert _source_dataset_id(survivor.json()) is None  # link severed

    def test_output_dataset_deletable_directly(self, client, ds_and_headers):
        """An experiment's result table can be deleted like any dataset; the
        experiment keeps its metrics but loses the stored table."""
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h,
        ).json()["id"]
        out_id = next(
            d["id"] for d in client.get("/api/datasets", headers=h).json()
            if _prov_kind(d) == "experiment_output" and f"exp #{exp_id}" in d["name"]
        )
        assert client.delete(f"/api/datasets/{out_id}", headers=h).status_code == 204
        exp = client.get(f"/api/experiments/{exp_id}", headers=h)
        assert exp.status_code == 200
        mcd = next(x for x in exp.json()["results"] if x["strategy"] == "mcd")
        assert mcd["gamma_after"] is not None  # metrics survive
        assert mcd["has_result"] is False      # stored table gone
        prev = client.get(f"/api/experiments/{exp_id}/results/mcd/preview", headers=h)
        assert prev.status_code == 404

    def test_delete_experiment_keeps_result_datasets(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h,
        ).json()["id"]
        result_id = next(
            d["id"] for d in client.get("/api/datasets", headers=h).json()
            if _prov_kind(d) == "experiment_output"
        )
        assert client.delete(f"/api/experiments/{exp_id}", headers=h).status_code == 204
        saved = client.get(f"/api/datasets/{result_id}", headers=h).json()
        assert saved["provenance"]["kind"] == "experiment_output"
        assert "MCD" in saved["name"]
        assert client.delete(f"/api/datasets/{result_id}", headers=h).status_code == 204

    def test_delete_parent_after_experiment_keeps_results(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h,
        ).json()["id"]
        result_id = next(
            d["id"] for d in client.get("/api/datasets", headers=h).json()
            if _prov_kind(d) == "experiment_output"
        )
        assert client.delete(f"/api/experiments/{exp_id}", headers=h).status_code == 204
        assert client.delete(f"/api/datasets/{ds_id}", headers=h).status_code == 204
        saved = client.get(f"/api/datasets/{result_id}", headers=h).json()
        assert saved["provenance"]["kind"] == "experiment_output"
        assert saved["provenance"]["source_dataset_id"] is None

    def test_experiment_creates_result_datasets(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h,
        ).json()["id"]
        datasets = client.get("/api/datasets", headers=h).json()
        result_ds = [d for d in datasets if _prov_kind(d) == "experiment_output"]
        assert len(result_ds) == 1
        assert _source_dataset_id(result_ds[0]) == ds_id
        assert "MCD" in result_ds[0]["name"]
        sources = client.get("/api/trees/sources", headers=h).json()
        assert any(s["dataset_id"] == result_ds[0]["id"] for s in sources)
        assert not any(s["kind"] == "result" for s in sources)

    def test_tree_metrics_preview(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd", "gd"]}, headers=h
        ).json()["id"]
        r = client.post(f"/api/experiments/{exp_id}/tree-metrics", headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["before"]["depth"] is not None
        assert body["strategies"]["mcd"]["nodes"] is not None
        exp = client.get(f"/api/experiments/{exp_id}", headers=h).json()
        assert exp["depth_before"] is None

    def test_result_preview_and_download(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd", "gd"]}, headers=h
        ).json()["id"]
        prev = client.get(f"/api/experiments/{exp_id}/results/mcd/preview", headers=h)
        assert prev.status_code == 200
        assert prev.json()["total"] > 0
        dl = client.get(f"/api/experiments/{exp_id}/results/mcd/csv", headers=h)
        assert dl.status_code == 200
        assert dl.headers["content-type"].startswith("text/csv")
        gd_dl = client.get(f"/api/experiments/{exp_id}/results/gd/csv", headers=h)
        assert gd_dl.status_code == 200
        assert gd_dl.headers["content-type"].startswith("text/csv")
        details = client.get(f"/api/experiments/{exp_id}/results/mcd/details/csv", headers=h)
        assert details.status_code == 200, details.text
        assert details.headers["content-type"].startswith("text/csv")
        assert "profile,decisions,outcome" in details.text

    def test_summary_csv(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h
        ).json()["id"]
        summary = client.get(f"/api/experiments/{exp_id}/csv", headers=h)
        assert summary.status_code == 200
        assert "gamma_before" in summary.text

    def test_save_result_tree(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h
        ).json()["id"]
        preview = client.post(f"/api/experiments/{exp_id}/results/mcd/tree-metrics", headers=h)
        assert preview.status_code == 200
        saved = client.post(f"/api/experiments/{exp_id}/results/mcd/tree/build", headers=h)
        assert saved.status_code == 200, saved.text
        body = saved.json()
        assert body["tree_run_id"] >= 1
        assert body["strategy"] == "mcd"
        assert "text" in body

    def test_tree_ownership_on_builtin(self, client):
        cars = next(d for d in client.get("/api/datasets").json() if d["name"] == "cars")
        h1 = register_and_login(client, "treeowner1@t.com")
        h2 = register_and_login(client, "treeowner2@t.com")
        r1 = client.post(
            "/api/trees/build",
            json={"kind": "dataset", "dataset_id": cars["id"], "algorithm": "id3"},
            headers=h1,
        )
        r2 = client.post(
            "/api/trees/build",
            json={"kind": "dataset", "dataset_id": cars["id"], "algorithm": "id3"},
            headers=h2,
        )
        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert r1.json()["tree_run_id"] != r2.json()["tree_run_id"]
        runs1 = client.get("/api/trees", headers=h1).json()
        runs2 = client.get("/api/trees", headers=h2).json()
        assert len([r for r in runs1 if r["table_name"] == "cars"]) == 1
        assert len([r for r in runs2 if r["table_name"] == "cars"]) == 1

    def test_delete_own_tree(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        built = client.post(
            "/api/trees/build",
            json={"kind": "dataset", "dataset_id": ds_id},
            headers=h,
        )
        tree_id = built.json()["tree_run_id"]
        assert client.delete(f"/api/trees/{tree_id}", headers=h).status_code == 204
        assert client.get(f"/api/trees/{tree_id}/text", headers=h).status_code == 404
        assert client.get("/api/trees", headers=h).json() == []

    def test_delete_tree_isolated(self, client, ds_and_headers):
        ds_id, h1 = ds_and_headers
        h2 = register_and_login(client, "treedelete2@t.com")
        tree_id = client.post(
            "/api/trees/build",
            json={"kind": "dataset", "dataset_id": ds_id},
            headers=h1,
        ).json()["tree_run_id"]
        assert client.delete(f"/api/trees/{tree_id}", headers=h2).status_code == 404

    def test_tree_sources_and_unified_build(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h
        ).json()["id"]

        sources = client.get("/api/trees/sources", headers=h).json()
        assert any(s["kind"] == "dataset" and s["dataset_id"] == ds_id for s in sources)
        ds_src = next(s for s in sources if s["kind"] == "dataset" and s["dataset_id"] == ds_id)
        assert ds_src["origin"]["type"] == "upload"
        assert ds_src["built_algorithms"] == []
        res_src = next(
            s for s in sources
            if s["origin"]["type"] == "experiment_result" and s["experiment_id"] == exp_id
        )
        assert res_src["kind"] == "dataset"
        assert res_src["origin"]["input_dataset_name"] == ds_src["table_name"]

        built = client.post(
            "/api/trees/build",
            json={"kind": "dataset", "dataset_id": ds_id},
            headers=h,
        )
        assert built.status_code == 200, built.text
        sources_after = client.get("/api/trees/sources", headers=h).json()
        ds_after = next(s for s in sources_after if s["dataset_id"] == ds_id)
        assert "id3" in ds_after["built_algorithms"]

        built = client.post(
            "/api/trees/build",
            json={"kind": "dataset", "dataset_id": res_src["dataset_id"]},
            headers=h,
        )
        assert built.status_code == 200, built.text
        body = built.json()
        assert body["tree_run_id"] >= 1
        assert "text" in body

    def test_tree_registry(self, client, ds_and_headers):
        """GET /api/trees lists saved dataset-linked runs only."""
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h
        ).json()["id"]
        client.post(f"/api/datasets/{ds_id}/tree/build", headers=h)
        client.post(f"/api/experiments/{exp_id}/results/mcd/tree/build", headers=h)

        rows = client.get("/api/trees", headers=h).json()
        assert len(rows) >= 2
        kinds = {r["source"]["kind"] for r in rows}
        assert kinds == {"dataset"}
        origins = {r["origin"]["type"] for r in rows}
        assert "upload" in origins
        assert "experiment_result" in origins

        stranger = register_and_login(client, "treestranger@t.com")
        other = client.get("/api/trees", headers=stranger).json()
        assert not [r for r in other if r["source"]["dataset_id"] == ds_id]

    def test_delete_experiment(self, client, ds_and_headers):
        ds_id, h = ds_and_headers
        exp_id = client.post(
            "/api/experiments", json={"dataset_id": ds_id, "strategies": ["mcd"]}, headers=h
        ).json()["id"]
        assert client.delete(f"/api/experiments/{exp_id}", headers=h).status_code == 204
        assert client.get(f"/api/experiments/{exp_id}", headers=h).status_code == 404
