# vocabweb

实现功能：
1) 网站骨架（FastAPI + Jinja2 + 静态资源）
2) 文档解析（PDF/Word → 纯文本并在网页显示）

## 本地运行
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
# 打开 http://127.0.0.1:8000
```

## 目录结构
```
vocabweb_steps1-2_fresh/
  app.py
  nlp.py
  requirements.txt
  templates/
    index.html
    result.html
  static/
    style.css
```
