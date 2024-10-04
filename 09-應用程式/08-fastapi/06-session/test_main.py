from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from main import app, get_db
from models import Base

# 設置測試用的 SQLite 數據庫
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 依賴注入：使用測試數據庫
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# 覆寫 FastAPI 中的依賴
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

def test_register_user(test_db):
    # 測試用戶註冊
    response = client.post("/register/", json={"username": "testuser", "password": "testpassword"})
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert "id" in data

def test_login_user(test_db):
    # 首先註冊一個用戶
    client.post("/register/", json={"username": "testuser", "password": "testpassword"})
    
    # 測試用戶登錄
    response = client.post("/login/", data={"username": "testuser", "password": "testpassword"})
    assert response.status_code == 200
    assert response.json() == {"message": "Login successful!"}

def test_login_user_invalid_credentials(test_db):
    # 測試使用無效憑據登錄
    response = client.post("/login/", data={"username": "wronguser", "password": "wrongpassword"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid username or password"}

def test_logout_user(test_db):
    # 首先註冊並登錄一個用戶
    client.post("/register/", json={"username": "testuser", "password": "testpassword"})
    client.post("/login/", data={"username": "testuser", "password": "testpassword"})
    
    # 測試登出
    response = client.post("/logout/")
    assert response.status_code == 200
    assert response.json() == {"message": "Logout successful!"}
