import requests

API_URL = "https://api-credit-score.onrender.com"

def test_home():
    response = requests.get(f"{API_URL}/")
    assert response.status_code == 200
    assert "API de scoring crédit" in response.json()["message"]

def test_upload_valid():
    files = {
        "application_test": open("tests/sample_data/application_test_sample.csv", "rb"),
        "bureau": open("tests/sample_data/bureau_sample.csv", "rb"),
        "previous_application": open("tests/sample_data/previous_application_sample.csv", "rb")
    }
    data = {"sk_id_curr": "102545"}
    response = requests.post(f"{API_URL}/upload", files=files, data=data)
    assert response.status_code == 200
    assert "predictions" in response.json()

def test_upload_sk_id_invalid():
    files = {
        "application_test": open("tests/sample_data/application_test_sample.csv", "rb"),
        "bureau": open("tests/sample_data/bureau_sample.csv", "rb"),
        "previous_application": open("tests/sample_data/previous_application_sample.csv", "rb")
    }
    data = {"sk_id_curr": "999999999"}  # ID inexistant
    response = requests.post(f"{API_URL}/upload", files=files, data=data)
    assert response.status_code == 400  # <- corrigé ici (était 404)
    assert "detail" in response.json()

def test_upload_missing_file():
    data = {"sk_id_curr": "102545"}
    response = requests.post(f"{API_URL}/upload", files={}, data=data)
    assert response.status_code == 422  # 422 attendu si fichier requis manquant
