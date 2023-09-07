import os
import glob
import uuid
import base64
import zipfile
import urllib.request
import logging
import boto3
import json
import httpx
import threading

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


def do_inference(model_cache_id="", model_url="", input_url="", input_content="", pitch=0):
    # 모델 다운로드
    model_id=""
    if model_cache_id:
        model_id=model_cache_id
        logger.info(f'model id cached: {model_cache_id}')
    else:
        model_url_bytes=model_url.encode('ascii')
        model_url_base64_bytes=base64.b64encode(model_url_bytes)
        model_id=model_url_base64_bytes.decode("ascii")
        logger.info(f'model id converted: {model_id}')

    model_file=f'{model_dir}/{model_id}.zip'
    if not os.path.isfile(model_file):
        logger.info(f'model downloading: {model_url}')
        urllib.request.urlretrieve(model_url, model_file)
    logger.info(f'model downloaded: {model_file}')

    # 모델 압축 풀기
    model_file_extracted_dir=f'{model_dir}/{model_id}'
    if os.path.isdir(model_file_extracted_dir):
        logger.info(f'model cache hitted: {model_file_extracted_dir}')
    else:
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
    input_file_id=uuid.uuid4()
    input_file=f'{input_dir}/{input_file_id}.wav'
    if input_content:
        decoded=base64.b64decode(input_content)
        with open(input_file, "wb") as f:
            f.write(decoded)
        logger.info(f'input file decoded: {model_index_file}')
    else:
        urllib.request.urlretrieve(input_url, input_file)
        logger.info(f'input file downloaded: {input_file}')

    # 합성
    output_file_id=uuid.uuid4()
    output_file=f'{output_dir}/{output_file_id}.wav'
    logger.info(f'inference started: {output_file}')
    vc_single(
        sid=0,
        input_audio_path=input_file,
        f0_up_key=pitch,
        f0_file=None,
        f0_method="crepe",
        file_index=model_index_file,
        file_index2="",
        output_path=output_file,
        model_path=model_pth_file,
    )
    logger.info(f'inference completed: {output_file}')

    with open(output_file, 'rb') as f:
      contents = f.read()
      return base64.b64encode(contents)

sqs = boto3.client('sqs', region_name='ap-northeast-2')

queue_url = 'https://sqs.ap-northeast-2.amazonaws.com/075389491675/rvc-inference.fifo'

class InferenceParams(BaseModel):
    request_id: str
    model_url: str
    input_url: str
    callback_url: str = ''
    pitch: Optional[int] = 0

@app.post("/inference/v2")
def inference_v2(params: InferenceParams):
    deduplication_id=uuid.uuid4()
    response = sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps({
            "request_id": params.request_id,
            "model_url": params.model_url,
            "input_url": params.input_url,
            "callback_url": params.callback_url,
            "pitch": params.pitch,
        }),
        MessageGroupId="inference_requests",
        MessageDeduplicationId=str(deduplication_id)
    )
    logger.info(f'sqs message sent: {response["MessageId"]}')
    return { "request_id": params.request_id }

def process_sqs_message(message):
    logger.info(f'sqs message received: {message["Body"]}')
    try:
        params=json.loads(message['Body'])
        request_id=params["request_id"]
        model_url=params["model_url"]
        input_url=params["input_url"]
        pitch=params["pitch"]
        callback_url=params["callback_url"]

        encoded = do_inference(
            model_url=model_url,
            input_url=input_url,
            pitch=pitch
        )
        # 콜백
        logger.info(f'success callback started: {callback_url}')
        httpx.post(callback_url, json={
            "result": "success",
            "request_id": request_id,
            "data": encoded.decode('utf8')
        }, timeout=30.0)
        logger.info(f'success callback completed: {callback_url}')
    except HTTPException as e:
        httpx.post(callback_url, json={
            "result": "failed",
            "request_id": request_id,
            "detail": e.detail
        })
        logger.info(f'failure callback completed: {callback_url}, {e.detail}')
    except Exception as e:
        httpx.post(callback_url, json={
            "result": "failed",
            "request_id": request_id,
            "detail": f'{e}'
        })
        logger.info(f'failure callback completed: {callback_url}, {e}')
    finally:
        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=message['ReceiptHandle']
        )
        logger.info(f'sqs message deleted: {message["ReceiptHandle"]}')


def poll_sqs_messages():
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10
        )
        logger.info(f'sqs message received')
        messages = response.get('Messages')
        if messages:
            for message in messages:
                try:
                    process_sqs_message(message)
                except:
                    pass

@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=poll_sqs_messages, daemon=True)
    thread.start()

class InferenceSyncRequest(BaseModel):
    model_cache_id: str
    model_url: str
    input_content: str
    pitch: Optional[int] = 0

@app.post("/inference/sync")
def inference(request: InferenceSyncRequest):
    try:
        encoded=do_inference(
            model_cache_id=request.model_cache_id,
            model_url=request.model_url,
            input_content=request.input_content,
            pitch=request.pitch
        )
        return {"data": encoded}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}')

@app.post("/inference/test")
def inference_test():
    with open('./mock.wav', 'rb') as f:
      contents = f.read()
      return {"data": base64.b64encode(contents)}
