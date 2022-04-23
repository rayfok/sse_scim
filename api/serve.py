import logging

from app.parser import parse_pdf

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("uvicorn")

app = FastAPI()


@app.post("/api/parse")
def upload_paper_ui(request: Request, file: UploadFile = File(...)):

    parsed = parse_pdf(file.filename)

    return JSONResponse(content=parsed, status_code=200)
