# -*- coding: utf-8 -*-
from typing import Annotated
from fastapi import APIRouter, HTTPException, UploadFile, status, Form
from models.OCRModel import *
from models.RestfulModel import *
# from paddleocr import PaddleOCR
# from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
import requests
import os
import json


OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "japan")

router = APIRouter(prefix="/ocr", tags=["OCR"])

imageReader = ImageReader()

@router.post('/detect-by-file', response_model=RestfulModel, summary="detect text box in file")
async def detect_by_file(file: UploadFile):
    print("Running mode: ", mode)
    result = {}
    if file.filename.endswith((".jpg", ".jpeg",".png")):
        file_data = file.file
        file_bytes = file_data.read()
        result = imageReader.DetectTextBox(file_bytes, mode)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only upload file .jpg, .jpeg or .png"
        )

    return Response(json.dumps(result))
@router.post('/detect_and_recognition-by-file', response_model=RestfulModel, summary="detect text box in file")
async def detect_by_file(file: UploadFile):
    # print("Running mode: ", mode)
    result = {}
    if file.filename.endswith((".jpg", ".jpeg",".png")):
        file_data = file.file
        file_bytes = file_data.read()
        result = imageReader.ReadImageWithMode(file_bytes, mode = None,infos=None)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only upload file .jpg, .jpeg or .png"
        )

    return Response(result, media_type="image/png")

@router.post('/read-file-with-position', response_model=RestfulModel, summary="read file at some position")
async def read_file_with_position(file: UploadFile, positions: str=Form(), infos: str=Form()):
    items = json.loads(positions)
    configs = json.loads(infos)
    if file.filename.endswith((".jpg", ".jpeg",".png")):
        file_data = file.file
        file_bytes = file_data.read()
        output_file_bytes = file_bytes
        output_file_bytes = imageReader.ReadImageWithPos(file_bytes, configs, items)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only upload file .jpg, .jpeg or .png"
        )
    # return Response(json.dumps(items))
    return Response(output_file_bytes, media_type="image/png")

# @router.post('/read-file-with-mode', response_model=RestfulModel, summary="detect and read file with position mode")
@router.post('/read-file-with-mode', response_model=RestfulModel, summary="detect and read file with position mode")
async def read_file_with_mode(file: UploadFile, mode: Annotated[str, Form()], infos: Annotated[str, Form()]):
    if file.filename.endswith((".jpg", ".jpeg", ".png")):
        file_data = file.file
        file_bytes = file_data.read()
        # output_file_bytes = file_bytes
        output_file_bytes = imageReader.ReadImageWithMode(file_bytes, mode, infos)
        # result = imageReader.ReadImageWithMode(file_bytes, mode)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only upload file .jpg, .jpeg or .png"
        )
    # return Response(json.dumps(result))
    # return Response(json.dumps(items))
    return Response(output_file_bytes, media_type="image/png")