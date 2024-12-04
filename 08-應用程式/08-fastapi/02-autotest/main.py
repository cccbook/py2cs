# https://chatgpt.com/c/66ff5593-249c-8012-be08-e0c69b6bb55f
from fastapi import FastAPI
from pydantic import BaseModel

# 建立 FastAPI 實例
app = FastAPI()

# 定義數據模型
class Item(BaseModel):
    name: str
    price: float
    description: str = None
    tax: float = None

# 定義 GET 請求的路由
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# 定義 GET 請求的路由，接受參數
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# 定義 POST 請求的路由
@app.post("/items/")
def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

# 啟動 FastAPI 伺服器的方法通常是使用命令：`uvicorn main:app --reload`
