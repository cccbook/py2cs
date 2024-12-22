from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# 靜態文件與模板設置
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 模擬資料庫
articles = []

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "articles": articles})

@app.get("/article/create")
def create_article_form(request: Request):
    return templates.TemplateResponse("create.html", {"request": request})

@app.post("/article/create")
def create_article(title: str = Form(...), content: str = Form(...)):
    article_id = len(articles) + 1
    articles.append({"id": article_id, "title": title, "content": content})
    return RedirectResponse("/", status_code=302)

@app.get("/article/{article_id}")
def read_article(request: Request, article_id: int):
    article = next((article for article in articles if article["id"] == article_id), None)
    if not article:
        return RedirectResponse("/", status_code=404)
    return templates.TemplateResponse("detail.html", {"request": request, "article": article})
