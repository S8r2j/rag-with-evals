import io

import pytest
from httpx import AsyncClient

from app.storage import LocalStorage
from tests.conftest import register_and_login

PDF_BYTES = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"


def pdf_upload(content: bytes = PDF_BYTES, name: str = "report.pdf") -> dict:
    return {"file": (name, io.BytesIO(content), "application/pdf")}


async def test_upload_pdf(client: AsyncClient, auth_headers: dict, storage: LocalStorage) -> None:
    response = await client.post("/api/v1/documents", files=pdf_upload(), headers=auth_headers)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["size_bytes"] == len(PDF_BYTES)
    assert body["status"] == "pending"
    assert len(list(storage.root.rglob("*.pdf"))) == 1


@pytest.mark.parametrize(
    ("name", "content", "expected_type"),
    [
        ("notes.txt", "Plain text with unicode: नमस्ते".encode(), "text/plain"),
        ("readme.md", b"# Title\n\nSome markdown.", "text/markdown"),
        (
            "letter.docx",
            b"PK\x03\x04" + b"\x00" * 64,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    ],
)
async def test_upload_other_supported_types(
    client: AsyncClient, auth_headers: dict, name: str, content: bytes, expected_type: str
) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": (name, io.BytesIO(content), "application/octet-stream")},
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    assert response.json()["content_type"] == expected_type


async def test_unsupported_extension_is_rejected(client: AsyncClient, auth_headers: dict) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("tool.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
        headers=auth_headers,
    )
    assert response.status_code == 415


async def test_content_that_does_not_match_the_extension_is_rejected(
    client: AsyncClient, auth_headers: dict
) -> None:
    response = await client.post(
        "/api/v1/documents",
        files=pdf_upload(content=b"MZ\x90\x00 not really a pdf"),
        headers=auth_headers,
    )
    assert response.status_code == 415
    assert "do not match" in response.json()["error"]["message"]


async def test_binary_disguised_as_text_is_rejected(
    client: AsyncClient, auth_headers: dict
) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("data.txt", io.BytesIO(b"\x00\x01\x02binary"), "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 415


async def test_file_over_the_size_limit_is_rejected(
    client: AsyncClient, auth_headers: dict, storage: LocalStorage
) -> None:
    too_big = PDF_BYTES + b"0" * (1024 * 1024)  # the test limit is 1 MB
    response = await client.post(
        "/api/v1/documents", files=pdf_upload(content=too_big), headers=auth_headers
    )
    assert response.status_code == 413
    assert not list(storage.root.rglob("*")) or not list(storage.root.rglob("*.pdf"))


async def test_empty_file_is_rejected(client: AsyncClient, auth_headers: dict) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 400


async def test_duplicate_upload_is_rejected(client: AsyncClient, auth_headers: dict) -> None:
    first = await client.post("/api/v1/documents", files=pdf_upload(), headers=auth_headers)
    assert first.status_code == 201
    second = await client.post(
        "/api/v1/documents", files=pdf_upload(name="copy.pdf"), headers=auth_headers
    )
    assert second.status_code == 409
    assert first.json()["id"] in second.json()["error"]["message"]


async def test_same_file_can_be_uploaded_by_different_users(
    client: AsyncClient, auth_headers: dict
) -> None:
    other = await register_and_login(client, "bob@example.com")
    assert (
        await client.post("/api/v1/documents", files=pdf_upload(), headers=auth_headers)
    ).status_code == 201
    assert (
        await client.post("/api/v1/documents", files=pdf_upload(), headers=other)
    ).status_code == 201


async def test_path_in_filename_is_stripped(client: AsyncClient, auth_headers: dict) -> None:
    response = await client.post(
        "/api/v1/documents", files=pdf_upload(name="../../etc/passwd.pdf"), headers=auth_headers
    )
    assert response.status_code == 201
    assert response.json()["filename"] == "passwd.pdf"


async def test_upload_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/documents", files=pdf_upload())
    assert response.status_code == 401


async def test_list_is_paginated_newest_first(client: AsyncClient, auth_headers: dict) -> None:
    for i in range(3):
        content = PDF_BYTES + f"% version {i}\n".encode()
        response = await client.post(
            "/api/v1/documents", files=pdf_upload(content, f"doc{i}.pdf"), headers=auth_headers
        )
        assert response.status_code == 201

    page = await client.get("/api/v1/documents?limit=2&offset=0", headers=auth_headers)
    assert page.status_code == 200
    body = page.json()
    assert body["total"] == 3
    assert [d["filename"] for d in body["items"]] == ["doc2.pdf", "doc1.pdf"]

    rest = (await client.get("/api/v1/documents?limit=2&offset=2", headers=auth_headers)).json()
    assert [d["filename"] for d in rest["items"]] == ["doc0.pdf"]


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1"])
async def test_invalid_pagination_is_rejected(
    client: AsyncClient, auth_headers: dict, query: str
) -> None:
    response = await client.get(f"/api/v1/documents?{query}", headers=auth_headers)
    assert response.status_code == 422


async def test_users_cannot_see_each_others_documents(
    client: AsyncClient, auth_headers: dict
) -> None:
    created = await client.post("/api/v1/documents", files=pdf_upload(), headers=auth_headers)
    document_id = created.json()["id"]
    intruder = await register_and_login(client, "mallory@example.com")

    assert (
        await client.get(f"/api/v1/documents/{document_id}", headers=intruder)
    ).status_code == 404
    assert (await client.get("/api/v1/documents", headers=intruder)).json()["total"] == 0
    assert (
        await client.delete(f"/api/v1/documents/{document_id}", headers=intruder)
    ).status_code == 404


async def test_get_and_delete_document(
    client: AsyncClient, auth_headers: dict, storage: LocalStorage
) -> None:
    created = await client.post("/api/v1/documents", files=pdf_upload(), headers=auth_headers)
    document_id = created.json()["id"]

    fetched = await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)
    assert fetched.status_code == 200
    assert fetched.json()["id"] == document_id

    deleted = await client.delete(f"/api/v1/documents/{document_id}", headers=auth_headers)
    assert deleted.status_code == 204
    assert (
        await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)
    ).status_code == 404
    assert not list(storage.root.rglob("*.pdf"))


async def test_unknown_document_returns_404(client: AsyncClient, auth_headers: dict) -> None:
    response = await client.get(
        "/api/v1/documents/00000000-0000-0000-0000-000000000000", headers=auth_headers
    )
    assert response.status_code == 404
