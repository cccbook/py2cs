from fastapi import FastAPI, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models, schemas, utils

app = FastAPI()

# 創建數據庫表
models.Base.metadata.create_all(bind=engine)

# 依賴注入: 獲取數據庫 session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 用戶註冊
@app.post("/register/", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    hashed_password = utils.hash_password(user.password)  # 加密密碼
    db_user = models.User(username=user.username, password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# 用戶登錄
@app.post("/login/")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None or not utils.verify_password(password, user.password):  # 驗證密碼
        raise HTTPException(status_code=400, detail="Invalid username or password")
    return {"message": "Login successful!"}

# 用戶登出（這裡只是一個示範，實際登出邏輯需要更具體的實現）
@app.post("/logout/")
def logout():
    return {"message": "Logout successful!"}
