from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# 創建 FastAPI 應用實例
app = FastAPI()

# 設定模板目錄
templates = Jinja2Templates(directory="templates")

# 若有靜態文件，如 CSS 或 JS，則可設置靜態文件路徑
app.mount("/static", StaticFiles(directory="static"), name="static")

# 定義一個路由來渲染 HTML 模板
@app.get("/")
async def read_item(request: Request):
    # 使用模板並傳遞數據
    return templates.TemplateResponse("index.html", {"request": request, "title": "Welcome to FastAPI!"})
