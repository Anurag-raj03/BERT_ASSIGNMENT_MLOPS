import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.parametrize("sentence, expected_label", [
    ("The patient denies chest pain.", "ABSENT"),
    ("He has a history of hypertension.", "PRESENT"),
    ("If the patient experiences dizziness, reduce the dosage.", "PRESENT"),
    ("No signs of pneumonia were observed.", "ABSENT")
])
def test_predict(client, sentence, expected_label):
    response = client.post("/predict", json={"sentence": sentence})
    assert response.status_code == 200
    
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert data["label"] == expected_label
    assert isinstance(data["score"], float)
    assert 0.0 <= data["score"] <= 1.0

def test_predict_batch(client):
    sentences = [
        "The patient denies chest pain.",
        "He has a history of hypertension."
    ]
    response = client.post("/predict_batch", json={"sentences": sentences})
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["label"] == "ABSENT"
    assert data["predictions"][1]["label"] == "PRESENT"