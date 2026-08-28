"""API surface without LLM/embedding calls: uploads, session isolation, eviction,
validation bounds. Ordered so the eviction flood runs last."""
import io

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app, MAX_SESSIONS, MAX_UPLOAD_BYTES


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def make_pdf(sentence: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), (sentence + " ") * 30, fontsize=11)
    return doc.tobytes()


def upload(client, session_id, marker, filename=None):
    return client.post(
        "/api/upload-pdf",
        files={"file": (filename or f"{marker}.pdf", io.BytesIO(make_pdf(f"codeword {marker}.")), "application/pdf")},
        headers={"X-Session-Id": session_id},
    )


def test_txt_and_md_uploads(client):
    md = b"# Notes\n\nThe crested lark nests in open farmland.\n\nIt sings in flight."
    r = client.post("/api/upload-pdf", files={"file": ("n.md", io.BytesIO(md), "text/markdown")},
                    headers={"X-Session-Id": "api-md"})
    assert r.status_code == 200 and "crested lark" in r.json()["preview"]
    r = client.post("/api/chunk", json={"strategy": "fixed", "chunk_size": 300},
                    headers={"X-Session-Id": "api-md"})
    assert r.status_code == 200


def test_unsupported_extension_rejected(client):
    r = client.post("/api/upload-pdf", files={"file": ("x.docx", io.BytesIO(b"z"), "application/octet-stream")})
    assert r.status_code == 400


def test_oversized_upload_rejected(client):
    big = b"%PDF-1.4 " + b"x" * (MAX_UPLOAD_BYTES + 100)
    r = client.post("/api/upload-pdf", files={"file": ("big.pdf", io.BytesIO(big), "application/pdf")})
    assert r.status_code == 413


def test_corrupt_pdf_returns_500_without_temp_leak(client, tmp_path):
    import glob, tempfile
    before = set(glob.glob(tempfile.gettempdir() + "/*.pdf"))
    r = client.post("/api/upload-pdf", files={"file": ("bad.pdf", io.BytesIO(b"not a pdf"), "application/pdf")})
    assert r.status_code == 500
    assert set(glob.glob(tempfile.gettempdir() + "/*.pdf")) - before == set()


def test_session_isolation(client):
    upload(client, "alice", "ALPHA")
    upload(client, "bob", "BRAVO")
    ra = client.post("/api/chunk", json={"strategy": "fixed", "chunk_size": 300}, headers={"X-Session-Id": "alice"})
    rb = client.post("/api/chunk", json={"strategy": "fixed", "chunk_size": 300}, headers={"X-Session-Id": "bob"})
    alice_text = " ".join(c["text"] for c in ra.json()["chunks_preview"])
    bob_text = " ".join(c["text"] for c in rb.json()["chunks_preview"])
    assert "ALPHA" in alice_text and "BRAVO" not in alice_text
    assert "BRAVO" in bob_text and "ALPHA" not in bob_text


def test_query_param_validation_bounds(client):
    r = client.post("/api/query-agentic", json={"question": "x", "top_k": 99})
    assert r.status_code == 422
    r = client.post("/api/query-agentic", json={"question": "x", "relevance_threshold": 1.5})
    assert r.status_code == 422


def test_fresh_session_gets_plain_400(client):
    r = client.post("/api/chunk", json={"strategy": "fixed"}, headers={"X-Session-Id": "never-seen"})
    assert r.status_code == 400
    assert "No document uploaded" in r.json()["detail"]


def test_zz_eviction_yields_410(client):
    """Runs last: floods the LRU, evicting earlier sessions."""
    upload(client, "evictee", "GAMMA")
    for i in range(MAX_SESSIONS + 5):
        client.post("/api/chunk", json={"strategy": "fixed"}, headers={"X-Session-Id": f"flood-{i}"})
    r = client.post("/api/chunk", json={"strategy": "fixed"}, headers={"X-Session-Id": "evictee"})
    assert r.status_code == 410
    assert "session expired" in r.json()["detail"]
    # Re-uploading recovers
    upload(client, "evictee", "GAMMA")
    r = client.post("/api/chunk", json={"strategy": "fixed"}, headers={"X-Session-Id": "evictee"})
    assert r.status_code == 200


def test_describe_document_samples_across_whole_document():
    """Regression: sampling only the head of a long document describes chapter one, and a
    router shown just that rejects on-topic questions about every later chapter."""
    from app.main import describe_document

    body = ("WATER AND ELECTROLYTES " * 200 + "PROTEIN REQUIREMENTS " * 200
            + "FAT SOLUBLE VITAMINS " * 200 + "ENERGY BALANCE " * 200)
    desc = describe_document({"pdf_filename": "nutrition.pdf", "pdf_text": body})

    assert "nutrition.pdf" in desc
    # Every section must be represented, not just the opening one.
    for topic in ("WATER", "PROTEIN", "VITAMINS", "ENERGY"):
        assert topic in desc, f"{topic} missing, sampling collapsed to the head"
    assert "fraction of the whole" in desc  # the router is told the sample is partial


def test_describe_document_handles_short_and_empty_inputs():
    from app.main import describe_document

    short = describe_document({"pdf_filename": "note.pdf", "pdf_text": "  a   short   note.  "})
    assert "a short note." in short          # whitespace collapsed
    assert describe_document({"pdf_text": "", "pdf_filename": "x.pdf"}) == ""
    assert describe_document({}) == ""
