"""Tests for Vertex AI authentication modes (Service Account vs. ADC / CLI)."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from utils import llm_client
from utils.llm_client import (
    GeminiVertexClient,
    GeminiHybridClient,
    resolve_google_vertex_auth_mode,
)


def test_resolve_google_vertex_auth_mode_explicit():
    assert resolve_google_vertex_auth_mode(auth_mode="adc") == "adc"
    assert resolve_google_vertex_auth_mode(auth_mode="cli") == "adc"
    assert resolve_google_vertex_auth_mode(auth_mode="application_default_credentials") == "adc"
    assert resolve_google_vertex_auth_mode(auth_mode="service_account") == "service_account"
    assert resolve_google_vertex_auth_mode(auth_mode="sa") == "service_account"
    assert resolve_google_vertex_auth_mode(auth_mode="json") == "service_account"

    with pytest.raises(ValueError, match="Invalid Google Vertex auth mode"):
        resolve_google_vertex_auth_mode(auth_mode="unsupported_mode")


def test_resolve_google_vertex_auth_mode_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_AUTH_MODE", "adc")
    assert resolve_google_vertex_auth_mode() == "adc"

    monkeypatch.setenv("GOOGLE_AUTH_MODE", "service_account")
    assert resolve_google_vertex_auth_mode() == "service_account"

    # Specific override
    monkeypatch.setenv("GOOGLE_VERTEX_AUTH_MODE", "cli")
    assert resolve_google_vertex_auth_mode() == "adc"


def test_resolve_google_vertex_auth_mode_auto(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_AUTH_MODE", raising=False)
    monkeypatch.delenv("GOOGLE_VERTEX_AUTH_MODE", raising=False)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_FILE", raising=False)

    # When service account files are explicitly supplied
    assert (
        resolve_google_vertex_auth_mode(service_account_files=[tmp_path / "sa.json"])
        == "service_account"
    )

    # When GOOGLE_SERVICE_ACCOUNT_FILE is set
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", str(tmp_path / "sa.json"))
    assert resolve_google_vertex_auth_mode() == "service_account"

    # When GOOGLE_SERVICE_ACCOUNT_FILE is set to 'adc'
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", "adc")
    assert resolve_google_vertex_auth_mode() == "adc"


def test_vertex_client_adc_initialization(monkeypatch):
    fake_creds = SimpleNamespace(
        quota_project_id="test-quota-project",
        project_id="creds-proj",
    )
    captured_client_args = {}

    def fake_default(scopes=None):
        return fake_creds, "default-proj"

    def fake_genai_client(*args, **kwargs):
        captured_client_args.update(kwargs)
        client_mock = MagicMock()
        return client_mock

    monkeypatch.setattr("google.auth.default", fake_default)
    monkeypatch.setattr("google.genai.Client", fake_genai_client)
    monkeypatch.setenv("GOOGLE_AUTH_MODE", "adc")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("CLOUDSDK_CORE_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)

    client = GeminiVertexClient(location="us-central1")

    assert client.auth_mode == "adc"
    assert client.service_account_file is None
    assert client.service_account_files == []
    assert client.project == "test-quota-project"
    assert client.location == "us-central1"
    assert captured_client_args.get("vertexai") is True
    assert captured_client_args.get("project") == "test-quota-project"
    assert captured_client_args.get("location") == "us-central1"
    assert captured_client_args.get("credentials") is fake_creds


def test_vertex_client_adc_chat_success(monkeypatch):
    fake_creds = SimpleNamespace(quota_project_id="test-project")
    monkeypatch.setattr("google.auth.default", lambda scopes=None: (fake_creds, "test-project"))

    fake_response = SimpleNamespace(
        text="ADC response text",
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
    )
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_genai_client)

    client = GeminiVertexClient(auth_mode="adc", project="test-project")
    response = client.chat([{"role": "user", "content": "Hello via ADC"}])

    assert response.content == "ADC response text"
    assert response.model == "gemini-2.5-flash"
    assert mock_genai_client.models.generate_content.called


def test_vertex_client_adc_missing_project_error(monkeypatch):
    fake_creds = SimpleNamespace(quota_project_id=None, project_id=None)
    monkeypatch.setattr("google.auth.default", lambda scopes=None: (fake_creds, None))
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("CLOUDSDK_CORE_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)

    with pytest.raises(ValueError, match="Vertex Gemini in ADC mode requires a project ID"):
        GeminiVertexClient(auth_mode="adc", project=None)


def test_vertex_client_adc_missing_credentials_error(monkeypatch):
    class DefaultCredentialsError(Exception):
        pass

    def fake_default(scopes=None):
        raise DefaultCredentialsError("ADC not found")

    monkeypatch.setattr("google.auth.default", fake_default)
    with pytest.raises(RuntimeError, match="gcloud auth application-default login"):
        GeminiVertexClient(auth_mode="adc", project="proj-123")


def test_gemini_hybrid_client_with_adc(monkeypatch):
    fake_creds = SimpleNamespace(quota_project_id="hybrid-proj")
    monkeypatch.setattr("google.auth.default", lambda scopes=None: (fake_creds, "hybrid-proj"))

    fake_vertex_response = SimpleNamespace(
        text="hybrid vertex text",
        usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=4),
    )
    mock_vertex_genai = MagicMock()
    mock_vertex_genai.models.generate_content.return_value = fake_vertex_response

    mock_studio_genai = MagicMock()

    def fake_genai_factory(**kwargs):
        if kwargs.get("vertexai") or kwargs.get("enterprise"):
            return mock_vertex_genai
        return mock_studio_genai

    monkeypatch.setattr("google.genai.Client", fake_genai_factory)
    monkeypatch.setenv("GOOGLE_AUTH_MODE", "adc")

    hybrid_client = GeminiHybridClient(
        api_keys=["studio-key-1"],
        project="hybrid-proj",
    )

    assert hybrid_client.auth_mode == "adc"
    assert hybrid_client.vertex_client.auth_mode == "adc"
    assert hybrid_client.active_transport == "vertex"

    response = hybrid_client.chat([{"role": "user", "content": "hi"}])
    assert response.content == "hybrid vertex text"
