"""Tests for the FastAPI layer (api/index.py): the NEW glue only.

The engine (extraction/verification) is covered by test_verification.py; here we
mock extract_fields and test what the API adds: upload validation, application
parsing, mode selection, error mapping, and the engine invariants the API must
preserve (the extractor never sees application data).

Run:  pytest tests/test_api.py
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import api.index as api_index
from extraction import ExtractionError, _coerce

client = TestClient(api_index.app)

# Minimal bytes that pass the magic-byte sniff (the engine never decodes images
# locally — the vision model does — so a signature plus filler is enough here).
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def compliant_extraction() -> dict:
    """A clean spirits read, built through the engine's own _coerce so the
    fixture can never drift from the extraction schema."""
    return _coerce({
        "beverage_type": "spirits",
        "brand_name": {"present": True, "value": "OLD TOM RESERVE", "confidence": "high"},
        "class_type": {"present": True, "value": "Kentucky Straight Bourbon Whiskey",
                       "confidence": "high"},
        "alcohol_content": {"present": True, "value": "45% Alc./Vol. (90 Proof)",
                            "abv_percent": 45.0, "proof": 90.0, "confidence": "high"},
        "net_contents": {"present": True, "value": "750 mL", "confidence": "high"},
        "name_and_address": {"present": True,
                             "value": "DISTILLED AND BOTTLED BY OLD TOM DISTILLERY, BARDSTOWN, KY",
                             "confidence": "high"},
        "government_warning": {
            "present": True,
            "text": ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
                     "not drink alcoholic beverages during pregnancy because of the risk of "
                     "birth defects. (2) Consumption of alcoholic beverages impairs your "
                     "ability to drive a car or operate machinery, and may cause health "
                     "problems."),
            "header_all_caps": True,
            "header_bold": True, "header_bold_confidence": "high",
            "header_bold_basis": "header strokes visibly thicker than body",
            "body_bold": False, "body_bold_confidence": "high",
            "confidence": "high",
        },
        "additional_statements": [],
        "image_quality_notes": None,
    })


MATCHING_APPLICATION = {
    "brand_name": "Old Tom Reserve",
    "class_type": "Kentucky Straight Bourbon Whiskey",
    "alcohol_content": "45% Alc./Vol.",
    "net_contents": "750 mL",
    "name_and_address": "Distilled and Bottled by Old Tom Distillery, Bardstown, KY",
}


@pytest.fixture
def mock_extract(monkeypatch):
    """Replace the real model call; record what the API passes to the extractor."""
    calls = []

    def fake_extract(images, media_type="image/png"):
        calls.append(images)
        return compliant_extraction()

    monkeypatch.setattr(api_index, "extract_fields", fake_extract)
    return calls


def post_verify(files=None, application=None):
    files = files if files is not None else [("images", ("front.png", PNG, "image/png"))]
    data = {"application": json.dumps(application)} if application is not None else {}
    return client.post("/api/py/verify", files=files, data=data)


# --- happy paths ---------------------------------------------------------------

def test_health():
    r = client.get("/api/py/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model"]


def test_rules_only_happy_path(mock_extract):
    r = post_verify()
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "rules_only"
    assert body["overall"] == "pass"
    assert body["beverage_type"] == "spirits"
    by_field = {f["field"]: f for f in body["fields"]}
    # rules-only screening for a non-wine product checks exactly these six
    assert set(by_field) == {"brand_name", "class_type", "alcohol_content",
                             "net_contents", "name_and_address", "government_warning"}
    assert by_field["government_warning"]["status"] == "pass"


def test_application_match_happy_path(mock_extract):
    r = post_verify(application=MATCHING_APPLICATION)
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "application_match"
    assert body["overall"] == "pass"
    by_field = {f["field"]: f for f in body["fields"]}
    assert by_field["brand_name"]["expected"] == MATCHING_APPLICATION["brand_name"]
    # verify() adds the import check (vs the six rules-only fields)
    assert "country_of_origin" in by_field


def test_two_images_read_together_as_one_product(mock_extract):
    r = post_verify(files=[("images", ("front.png", PNG, "image/png")),
                           ("images", ("back.jpg", JPEG, "image/jpeg"))])
    assert r.status_code == 200
    assert len(mock_extract) == 1          # ONE extraction call for the product
    assert len(mock_extract[0]) == 2       # both images in it


def test_blank_application_means_rules_only(mock_extract):
    """The independent-witness invariant at the API boundary: an all-blank form
    must screen rules-only, never silently 'match'."""
    r = post_verify(application={"brand_name": "  ", "net_contents": ""})
    assert r.status_code == 200
    assert r.json()["mode"] == "rules_only"


def test_extractor_never_sees_application(mock_extract):
    """Engine invariant: the extractor is blind to expected values."""
    post_verify(application=MATCHING_APPLICATION)
    (pairs,) = mock_extract
    blob = repr(pairs)
    for value in MATCHING_APPLICATION.values():
        assert value not in blob


def test_media_type_is_sniffed_not_trusted(mock_extract):
    """A PNG mislabeled as JPEG by the client is sent to the engine as PNG."""
    post_verify(files=[("images", ("front.jpg", PNG, "image/jpeg"))])
    (pairs,) = mock_extract
    assert pairs[0][1] == "image/png"


# --- validation & error mapping --------------------------------------------------

def test_missing_images_field_is_400_envelope():
    r = client.post("/api/py/verify")
    assert r.status_code == 400
    assert r.json()["error"]["kind"] == "invalid_request"


def test_non_image_rejected():
    r = post_verify(files=[("images", ("label.txt", b"not an image at all", "text/plain"))])
    assert r.status_code == 400
    assert r.json()["error"]["kind"] == "unsupported_type"


def test_too_many_images_rejected():
    files = [("images", (f"img{i}.png", PNG, "image/png")) for i in range(3)]
    r = post_verify(files=files)
    assert r.status_code == 400
    assert r.json()["error"]["kind"] == "too_many_images"


def test_oversize_file_is_413_with_clear_message():
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (api_index.MAX_FILE_BYTES + 1)
    r = post_verify(files=[("images", ("huge.png", big, "image/png"))])
    assert r.status_code == 413
    assert r.json()["error"]["kind"] == "file_too_large"


def test_total_size_cap_is_413():
    nearly = b"\x89PNG\r\n\x1a\n" + b"\x00" * (api_index.MAX_TOTAL_BYTES // 2 + 1024)
    r = post_verify(files=[("images", ("a.png", nearly, "image/png")),
                           ("images", ("b.png", nearly, "image/png"))])
    assert r.status_code == 413
    assert r.json()["error"]["kind"] == "payload_too_large"


def test_malformed_application_json_is_400(mock_extract):
    r = client.post("/api/py/verify",
                    files=[("images", ("front.png", PNG, "image/png"))],
                    data={"application": "{not json"})
    assert r.status_code == 400
    assert r.json()["error"]["kind"] == "invalid_application"


def test_unknown_application_key_is_400(mock_extract):
    """extra='forbid': a drifted frontend payload fails loudly, not silently."""
    r = post_verify(application={"brand": "Old Tom"})
    assert r.status_code == 400
    assert r.json()["error"]["kind"] == "invalid_application"


def test_extraction_failure_maps_to_502(monkeypatch):
    def boom(images, media_type="image/png"):
        raise ExtractionError("the model returned an empty response")

    monkeypatch.setattr(api_index, "extract_fields", boom)
    r = post_verify()
    assert r.status_code == 502
    body = r.json()["error"]
    assert body["kind"] == "bad_response"
    # the envelope must carry the curated message, never the raw exception text
    assert body["message"] == api_index._FAILURE_RESPONSES["bad_response"][1]


def test_unexpected_crash_still_returns_error_envelope(monkeypatch):
    """A crash AFTER extraction (e.g. in verify()) must still produce the
    documented {error:{kind,message}} JSON envelope, not a text/plain 500."""
    monkeypatch.setattr(api_index, "extract_fields",
                        lambda images, media_type="image/png": compliant_extraction())
    monkeypatch.setattr(api_index, "verify_label_only",
                        lambda extracted: (_ for _ in ()).throw(RuntimeError("boom")))
    crashing_client = TestClient(api_index.app, raise_server_exceptions=False)
    r = crashing_client.post("/api/py/verify",
                             files=[("images", ("front.png", PNG, "image/png"))])
    assert r.status_code == 500
    body = r.json()["error"]
    assert body["kind"] == "unknown"
    assert body["message"] == api_index._FAILURE_RESPONSES["unknown"][1]


def test_extraction_runs_off_the_event_loop(monkeypatch):
    """The engine's OpenAI call is synchronous; the endpoint must run it in the
    threadpool so a slow read doesn't freeze concurrent requests (uvicorn dev,
    Vercel in-function concurrency). A blocking /verify is started, then /health
    is called while it is still in flight — health must answer first."""
    import asyncio
    import threading
    import time

    from httpx import ASGITransport, AsyncClient

    started = threading.Event()

    def slow_extract(images, media_type="image/png"):
        started.set()
        time.sleep(0.6)
        return compliant_extraction()

    monkeypatch.setattr(api_index, "extract_fields", slow_extract)

    async def scenario():
        transport = ASGITransport(app=api_index.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            verify_task = asyncio.create_task(
                ac.post("/api/py/verify",
                        files=[("images", ("front.png", PNG, "image/png"))]))
            await asyncio.to_thread(started.wait, 5)   # extraction is now in flight
            t0 = time.monotonic()
            health = await ac.get("/api/py/health")
            health_latency = time.monotonic() - t0
            verify = await verify_task
            return health.status_code, health_latency, verify.status_code

    health_status, health_latency, verify_status = asyncio.run(scenario())
    assert health_status == 200 and verify_status == 200
    assert health_latency < 0.5, (
        f"/health took {health_latency:.2f}s while an extraction was in flight — "
        "the model call is blocking the event loop")


def test_phantom_422_stripped_from_openapi():
    """Validation errors are remapped to 400, so the generated docs must not
    advertise FastAPI's default 422."""
    schema = api_index.app.openapi()
    verify_responses = schema["paths"]["/api/py/verify"]["post"]["responses"]
    assert "422" not in verify_responses
    assert "400" in verify_responses
