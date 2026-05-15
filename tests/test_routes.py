"""Unit tests for Flask routes (mocked analysis)."""

import pytest
from unittest.mock import patch

from app import create_app


@pytest.fixture()
def client():
    app = create_app(testing=True)
    with app.test_client() as c:
        yield c


class TestHomePage:
    def test_home_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Predictor Dashboard" in resp.data

    @patch("app.routes.analysis.server_is_available", return_value=True)
    def test_home_server_up(self, mock_srv, client):
        resp = client.get("/")
        assert b"Online" in resp.data

    @patch("app.routes.analysis.server_is_available", return_value=False)
    def test_home_server_down(self, mock_srv, client):
        resp = client.get("/")
        assert b"Offline" in resp.data


class TestErrorPages:
    @patch("app.routes.analysis.server_is_available", return_value=False)
    def test_explicit_shows_error(self, mock_srv, client):
        resp = client.get("/explicit")
        assert resp.status_code == 200
        assert b"Server Unavailable" in resp.data

    @patch("app.routes.analysis.server_is_available", return_value=False)
    def test_predictor_shows_error(self, mock_srv, client):
        resp = client.get("/predictor")
        assert resp.status_code == 200
        assert b"Server Unavailable" in resp.data

    @patch("app.routes.analysis.server_is_available", return_value=False)
    def test_comparison_shows_error(self, mock_srv, client):
        resp = client.get("/comparison")
        assert resp.status_code == 200
        assert b"Server Unavailable" in resp.data


class TestAPIEndpoints:
    def test_api_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "server_available" in data

    @patch("app.routes.analysis.server_is_available", return_value=False)
    def test_api_explicit_unavailable(self, mock_srv, client):
        resp = client.get("/api/explicit")
        assert resp.status_code == 503

    @patch("app.routes.analysis.server_is_available", return_value=False)
    def test_api_predictor_unavailable(self, mock_srv, client):
        resp = client.get("/api/predictor")
        assert resp.status_code == 503

    @patch("app.routes.analysis.server_is_available", return_value=False)
    def test_api_comparison_unavailable(self, mock_srv, client):
        resp = client.get("/api/comparison")
        assert resp.status_code == 503
