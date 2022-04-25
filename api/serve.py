import logging

from app.parser import get_sentences

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("uvicorn")

app = FastAPI()


@app.post("/api/parse")
def upload_paper_ui(request: Request, file: UploadFile = File(...)):

    parsed = get_sentences(file.filename)

    return JSONResponse(content=parsed, status_code=200)
