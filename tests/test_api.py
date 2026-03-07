from fastapi.testclient import TestClient
import pytest

from gas_energy_copilot.ai_copilot.core.application import initialize_app


@pytest.fixture
def test_client():
    return TestClient(initialize_app())


def test_read_main(test_client: TestClient) -> None:
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from AI Copilot!"}
