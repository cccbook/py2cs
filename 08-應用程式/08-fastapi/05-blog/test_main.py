from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from main import app, get_db
from models import Base
from database import Base

# 創建測試數據庫
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_blog.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 測試使用的數據庫依賴
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# 覆寫 FastAPI 中的依賴注入，使用測試數據庫
app.dependency_overrides[get_db] = override_get_db

# 創建測試用的數據庫表
Base.metadata.create_all(bind=engine)

client = TestClient(app)

@pytest.fixture
def test_db():
    # 每次測試前重置測試數據庫
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_create_post(test_db):
    # 測試創建文章 API
    response = client.post(
        "/posts/",
        json={"title": "Test Post", "content": "Test Content"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Post"
    assert data["content"] == "Test Content"
    assert "id" in data

def test_read_posts(test_db):
    # 創建一篇文章以便測試查詢功能
    client.post(
        "/posts/",
        json={"title": "Test Post", "content": "Test Content"},
    )
    
    # 測試查詢所有文章 API
    response = client.get("/posts/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Test Post"
    assert data[0]["content"] == "Test Content"

def test_read_single_post(test_db):
    # 創建一篇文章以便測試查詢單篇文章功能
    response = client.post(
        "/posts/",
        json={"title": "Single Post", "content": "Single Content"},
    )
    post_id = response.json()["id"]

    # 測試查詢單篇文章 API
    response = client.get(f"/posts/{post_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Single Post"
    assert data["content"] == "Single Content"
