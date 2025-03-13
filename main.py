from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import json
import tiktoken
from typing import List, Optional, Dict, Any
from fastapi.encoders import jsonable_encoder
from loguru import logger
from starlette.responses import StreamingResponse


app = FastAPI(title="Token Analyzer")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    try:
        body = body.decode() if body else None
    except Exception:
        body = "Unable to decode body"

    logger.info(f"🚀 Request: {request.method} {request.url}")
    logger.info(f"📡 Headers: {dict(request.headers)}")
    logger.info(f"📝 Body: {body}")

    # 让原始 response 先执行
    response = await call_next(request)
    
    # 重新读取 response body（避免消耗原始 body）
    response_body = [chunk async for chunk in response.body_iterator]
    response_text = b"".join(response_body).decode()

    logger.info(f"✅ Response: {response.status_code}")
    logger.info(f"📄 Response Body: {response_text}")

    # 重新构造 response（防止原始 body 被消耗）
    return StreamingResponse(iter(response_body), status_code=response.status_code, headers=dict(response.headers))
# 创建templates目录（如果不存在）
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# 创建static目录（如果不存在）
#s.makedirs("static/js", exist_ok=True)
# p.mount("/static", StaticFiles(directory="static"), name="static")

# 使用tiktoken初始化分词器
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 & GPT-3.5 使用的编码

class TokenRequest(BaseModel):
    text: str

class TokenResponse(BaseModel):
    tokens: List[int]
    token_count: int
    characters: int
    tokens_per_character: float
    decoded_text: str
    
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/tokenize/", response_model=TokenResponse)
async def tokenize(token_request: TokenRequest):
    text = token_request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    try:
        tokens = tokenizer.encode(text)
        
        # 不再解码单个token，只保留ID
        # 添加完整解码结果
        decoded_text = tokenizer.decode(tokens)
        
        response_data = TokenResponse(
            tokens=tokens,
            token_count=len(tokens),
            characters=len(text),
            tokens_per_character=len(tokens) / max(1, len(text)),
            decoded_text=decoded_text  # 添加完整解码结果
        )
        
        return JSONResponse(content=jsonable_encoder(response_data), media_type="application/json; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分词错误: {e}")


@app.post("/compare/", response_class=HTMLResponse)
async def compare(request: Request, text1: str = Form(...), text2: str = Form(...)):
    global tokenizer
    
    try:
        # 对两个输入文本进行分词
        tokens1 = tokenizer.encode(text1)
        tokens2 = tokenizer.encode(text2)
        
        # 准备结果
        result1 = {
            "tokens": tokens1,
            "token_count": len(tokens1),
            "characters": len(text1),
            "tokens_per_character": len(tokens1) / max(1, len(text1)),
        }
        
        result2 = {
            "tokens": tokens2,
            "token_count": len(tokens2),
            "characters": len(text2),
            "tokens_per_character": len(tokens2) / max(1, len(text2)),
        }
        
        # 计算差异
        comparison = {
            "token_count_diff": result2["token_count"] - result1["token_count"],
            "token_count_percentage": (result2["token_count"] / max(1, result1["token_count"]) * 100) - 100,
            "character_diff": result2["characters"] - result1["characters"],
            "character_percentage": (result2["characters"] / max(1, result1["characters"]) * 100) - 100,
            "efficiency_diff": result2["tokens_per_character"] - result1["tokens_per_character"]
        }
        
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "text1": text1,
                "text2": text2,
                "result1": result1, 
                "result2": result2,
                "comparison": comparison
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "text1": text1,
                "text2": text2,
                "error": f"分词错误: {e}"
            }
        )

@app.post("/bulk-analyze/")
async def bulk_analyze(file: UploadFile = File(...)):
    global tokenizer
    
    try:
        # 读取文件内容
        content = await file.read()
        text = content.decode("utf-8")
        
        # 按行分割
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # 处理每一行
        results = []
        for line in lines:
            try:
                tokens = tokenizer.encode(line)
                
                result = {
                    "text": line,
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "characters": len(line),
                    "tokens_per_character": len(tokens) / max(1, len(line))
                }
                
                results.append(result)
            except Exception as e:
                results.append({
                    "text": line,
                    "error": str(e)
                })
        
        # 计算汇总统计
        valid_results = [r for r in results if "error" not in r]
        
        if valid_results:
            summary = {
                "total_lines": len(results),
                "valid_lines": len(valid_results),
                "total_characters": sum(r["characters"] for r in valid_results),
                "total_tokens": sum(r["token_count"] for r in valid_results),
                "avg_tokens_per_character": sum(r["tokens_per_character"] for r in valid_results) / len(valid_results)
            }
        else:
            summary = {
                "total_lines": len(results),
                "valid_lines": 0,
                "total_characters": 0,
                "total_tokens": 0,
                "avg_tokens_per_character": 0
            }
        
        return {
            "results": results,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文件时出错: {e}")

@app.get("/tokenizer-info/")
async def get_tokenizer_info():
    global tokenizer
    
    # 获取基本分词器信息
    info = {
        "name": "cl100k_base (GPT-3.5 & GPT-4)",
        "type": "tiktoken"
    }
    
    # 尝试获取词汇量
    try:
        # tiktoken模型的词汇量通常是固定的
        info["vocab_size"] = tokenizer.n_vocab
    except Exception as e:
        info["vocab_size"] = "未知"
    
    # 获取一些样例token
    try:
        sample_tokens = {}
        # 一些常见的词语和字符
        samples = ["hello", "world", "AI", "tokenization", "你好", "世界"]
        
        for sample in samples:
            tokens = tokenizer.encode(sample)
            decoded = [tokenizer.decode([t]) for t in tokens]
            sample_tokens[sample] = {
                "token_ids": tokens,
                "decoded": [repr(d)[1:-1] for d in decoded]
            }
            
        info["sample_tokens"] = sample_tokens
    except Exception as e:
        pass
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)