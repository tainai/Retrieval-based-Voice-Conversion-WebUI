import os
import glob
import uuid
import base64
import zipfile
import urllib.request
import logging

from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from infer_cli import vc_single

now_dir=os.getcwd()
tmp_dir=os.path.join(now_dir, "tmp")
os.makedirs(tmp_dir, exist_ok=True)
model_dir=os.path.join(now_dir, "tmp/models")
os.makedirs(model_dir, exist_ok=True)
input_dir=os.path.join(now_dir, "tmp/inputs")
os.makedirs(input_dir, exist_ok=True)
output_dir=os.path.join(now_dir, "tmp/outputs")
os.makedirs(output_dir, exist_ok=True)

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RVC_SERVER")


@app.get("/")
def read_root():
    return {"ping": "pong"}


class InferenceRequest(BaseModel):
    model_url: str
    input_url: str
    pitch: Optional[int] = 0

@app.post("/inference")
def inference(request: InferenceRequest):
    # 모델 다운로드
    model_url=request.model_url
    logger.info(f'model downloading: {model_url}')
    model_url_bytes=model_url.encode('ascii')
    model_url_base64_bytes = base64.b64encode(model_url_bytes)
    model_id = model_url_base64_bytes.decode("ascii")
    logger.info(f'model id converted: {model_id}')
    model_file=f'{model_dir}/{model_id}.zip'
    if not os.path.isfile(model_file):
        urllib.request.urlretrieve(model_url, model_file)
    logger.info(f'model downloaded: {model_file}')

    # 모델 압축 풀기
    model_file_extracted_dir=f'{model_dir}/{model_id}'
    os.makedirs(model_file_extracted_dir, exist_ok=True)
    with zipfile.ZipFile(model_file, 'r') as zip_ref:
        zip_ref.extractall(model_file_extracted_dir)
    logger.info(f'model extracted: {model_file_extracted_dir}')

    # pth 파일 찾기
    model_pth_file=''
    for file in glob.glob(f'{model_file_extracted_dir}/**/*.pth', recursive=True):
        model_pth_file=file
    if not model_pth_file:
        raise HTTPException(status_code=404, detail="model pth file not found")
    logger.info(f'model pth file found: {model_pth_file}')

    # index 파일 찾기
    model_index_file=''
    for file in glob.glob(f'{model_file_extracted_dir}/**/*.index', recursive=True):
        model_index_file=file
    if not model_index_file:
        raise HTTPException(status_code=404, detail="model index file not found")
    logger.info(f'model index file found: {model_index_file}')

    # 오디오 다운로드
    input_url=request.input_url
    input_file_id=uuid.uuid4()
    input_file=f'{input_dir}/{input_file_id}.wav'
    urllib.request.urlretrieve(input_url, input_file)
    logger.info(f'input downloaded: {input_file}')

    # 합성
    output_file_id=uuid.uuid4()
    output_file=f'{output_dir}/{output_file_id}.wav'
    vc_single(
        sid=0,
        input_audio_path=input_file,
        f0_up_key=request.pitch,
        f0_file=None,
        f0_method="crepe",
        file_index=model_index_file,
        file_index2="",
        output_path=output_file,
        model_path=model_pth_file,
    )

    with open(output_file, 'rb') as f:
      contents = f.read()
      encoded=base64.b64encode(contents)
      return {"data": encoded}
