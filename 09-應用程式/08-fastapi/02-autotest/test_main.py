from fastapi.testclient import TestClient
from main import app

# 使用 FastAPI 的 TestClient 來創建測試客戶端
client = TestClient(app)

# 測試 GET /
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI!"}

# 測試 GET /items/{item_id}
def test_read_item():
    item_id = 1
    response = client.get(f"/items/{item_id}?q=test_query")
    assert response.status_code == 200
    assert response.json() == {"item_id": item_id, "q": "test_query"}

# 測試 POST /items/
def test_create_item():
    new_item = {
        "name": "Test Item",
        "price": 10.5,
        "description": "A test item",
        "tax": 1.5
    }
    response = client.post("/items/", json=new_item)
    assert response.status_code == 200
    assert response.json() == {
        "name": "Test Item",
        "price": 10.5,
        "description": "A test item",
        "tax": 1.5,
        "price_with_tax": 12.0
    }
