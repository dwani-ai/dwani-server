import argparse
import io
import os
from time import time
from typing import List

import tempfile
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.util import get_remote_address

#
import soundfile as sf
from pydub import AudioSegment



from config.logging_config import logger
from config.tts_config import SPEED, ResponseFormat, config as tts_config

#from models.asr import ASRManager
from models.gemma_llm import LLMManager
from models.translate import TranslateManager
from models.tts import TTSManager
#from models.vlm import VLMManager

from utils.auth import get_api_key, settings as auth_settings

from PIL import Image
import requests
import torch
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Form


class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-4b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"
    #allowed_origins: str = "http://localhost:7860,https://gaganyatri-llm-indic-server-vlm.hf.space"

    @field_validator("chat_rate_limit", "speech_rate_limit")
    def validate_rate_limit(cls, v):
        if not v.count("/") == 1 or not v.split("/")[0].isdigit():
            raise ValueError("Rate limit must be in format 'number/period' (e.g., '5/minute')")
        return v

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False,  
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
    #allow_methods=["GET", "POST"],
    #allow_headers=["X-API-Key", "Content-Type", "Accept"],
    #allow_origins=settings.allowed_origins.split(","),
    #allow_methods=["GET", "POST", "OPTIONS"],
    #allow_headers=["X-API-Key", "Content-Type", "Accept"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Initialize model managers with lazy loading
llm_manager = LLMManager(settings.llm_model_name)
tts_manager = TTSManager()
#vlm_manager = VLMManager()
#asr_manager = ASRManager()
translate_manager_eng_indic = TranslateManager("eng_Latn", "kan_Knda")
translate_manager_indic_eng = TranslateManager("kan_Knda", "eng_Latn")
translate_manager_indic_indic = TranslateManager("kan_Knda", "hin_Deva")


# Request/Response Models
class ChatRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()


class ChatResponse(BaseModel):
    response: str


class SpeechRequest(BaseModel):
    input: str
    voice: str
    model: str
    response_format: ResponseFormat = tts_config.response_format
    speed: float = SPEED

    @field_validator("input")
    def input_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Input cannot exceed 1000 characters")
        return v.strip()

    @field_validator("response_format")
    def validate_response_format(cls, v):
        supported_formats = [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV]
        if v not in supported_formats:
            raise ValueError(f"Response format must be one of {[fmt.value for fmt in supported_formats]}")
        return v


class TranscriptionResponse(BaseModel):
    text: str


class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]


class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str


class TranslationResponse(BaseModel):
    translations: List[str]


# Dependency for TranslateManager
def get_translate_manager(request: TranslationRequest = Body(...)):
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang
    if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
        return translate_manager_eng_indic
    elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
        return translate_manager_indic_eng
    elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
        return translate_manager_indic_indic
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid language combination: English to English translation is not supported.",
        )

# Endpoints
@app.get("/v1/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}


@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/unload_all_models")
async def unload_all_models(api_key: str = Depends(get_api_key)):
    try:
        logger.info("Starting to unload all models...")
        # vlm_manager.unload()
        # asr_manager.unload()
        llm_manager.unload()
        tts_manager.unload()
        translate_manager_eng_indic.unload()
        translate_manager_indic_eng.unload()
        translate_manager_indic_indic.unload()
        logger.info("All models unloaded successfully")
        return {"status": "success", "message": "All models unloaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

@app.post("/v1/load_all_models")
async def load_all_models(api_key: str = Depends(get_api_key)):
    try:
        logger.info("Starting to load all models...")
        #llm_manager.load()
        #tts_manager.load_model(tts_config.model)
        tts_manager.load_model(tts_config.model, compile_mode="reduce-overhead")
        #vlm_manager.load()
        #asr_manager.load()
        translate_manager_eng_indic.load()
        translate_manager_indic_eng.load()
        #translate_manager_indic_indic.load()
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")


@app.post("/v1/audio/speech")
@limiter.limit(settings.speech_rate_limit)
async def generate_audio(
    request: Request, speech_request: SpeechRequest = Body(...), api_key: str = Depends(get_api_key)
):
    if not speech_request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    logger.info(f"Speech request: input={speech_request.input[:50]}..., voice={speech_request.voice}")
    try:
        audio_arr = tts_manager.generate_audio(
            speech_request.input, speech_request.voice, speech_request.model, speech_request.speed
        )
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_arr, 24000, format=speech_request.response_format.value)
        audio_buffer.seek(0)
        headers = {
            "Content-Disposition": f"inline; filename=\"speech.{speech_request.response_format.value}\"",
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(
            audio_buffer, media_type=f"audio/{speech_request.response_format.value}", headers=headers
        )
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")


@app.post("/v1/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}")
    try:
        translated_prompt = translate_manager_indic_eng.translate(chat_request.prompt)
        logger.info(f"Translated prompt to English: {translated_prompt}")


        response = await llm_manager.generate(translated_prompt, settings.max_tokens)
        logger.info(f"Generated English response: {response}")
        translated_response = translate_manager_eng_indic.translate(response)
        logger.info(f"Translated response to Kannada: {translated_response}")
        return ChatResponse(response=translated_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

'''
@app.post("/v1/caption/")
async def caption_image(file: UploadFile = File(...), length: str = "normal"):
    image = Image.open(file.file)
    caption = vlm_manager.caption(image, length)
    return {"caption": caption}

@app.post("/v1/detect/")
async def detect_objects(file: UploadFile = File(...), object_type: str = "face"):
    image = Image.open(file.file)
    objects = vlm_manager.detect(image, object_type)
    return {"objects": objects}


@app.post("/v1/point/")
async def point_objects(file: UploadFile = File(...), object_type: str = "person"):
    image = Image.open(file.file)
    points = vlm_manager.point(image, object_type)
    return {"points": points}

'''
@app.post("/v1/visual_query/")
async def visual_query(image: UploadFile = File(...), query: str = Body(...)):
    #image = Image.open(file.file)

    try:
        # Construct the message structure
        answer = await llm_manager.vision_query(image, query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/v1/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query(..., enum=["kannada", "hindi", "tamil"]),  # Adjust supported languages as needed
    api_key: str = Depends(get_api_key),
    request: Request = None,
):
    logger.info(f"Request method: {request.method}, Headers: {request.headers}, Query: {request.query_params}")
    start_time = time()
    try:
        # Prepare the file for the external request
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}

        # Reroute to the external server
        external_url = f"https://gaganyatri-asr-indic-server-cpu.hf.space/transcribe/?language={language}"
        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"}
        )

        if response.status_code != 200:
            logger.error(f"External server error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"External server error: {response.text}")

        transcription = response.json().get("text", "")
        logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")
        return JSONResponse(content={"text": transcription})

    except Exception as e:
        logger.error(f"Error during transcription rerouting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


'''


@app.post("/v1/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query(..., enum=list(asr_manager.model_language.keys())),
    api_key: str = Depends(get_api_key),
    request: Request = None,  # Add for debugging
):
    logger.info(f"Request method: {request.method}, Headers: {request.headers}, Query: {request.query_params}")
    start_time = time()
    try:
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["wav", "mp3"]:
            logger.warning(f"Unsupported file format: {file_extension}")
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload a WAV or MP3 file."
            )

        file_content = await file.read()
        audio = (
            AudioSegment.from_mp3(io.BytesIO(file_content))
            if file_extension == "mp3"
            else AudioSegment.from_wav(io.BytesIO(file_content))
        )
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio.export(tmp_file.name, format="wav")
            tmp_file_path = tmp_file.name

        chunk_file_paths = asr_manager.split_audio(tmp_file_path)
        try:
            language_id = asr_manager.model_language.get(language, asr_manager.default_language)
            transcription = asr_manager.transcribe(chunk_file_paths, language_id)
            logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")
            return JSONResponse(content={"text": transcription})
        finally:
            for chunk_file_path in chunk_file_paths:
                if os.path.exists(chunk_file_path):
                    os.remove(chunk_file_path)
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            asr_manager.cleanup()
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/transcribe_batch/", response_model=BatchTranscriptionResponse)
async def transcribe_audio_batch(
    files: List[UploadFile] = File(...),
    language: str = Query(..., enum=list(asr_manager.model_language.keys())),
    api_key: str = Depends(get_api_key),
):
    start_time = time()
    all_transcriptions = []
    try:
        for file in files:
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["wav", "mp3"]:
                logger.warning(f"Unsupported file format: {file_extension}")
                raise HTTPException(
                    status_code=400, detail="Unsupported file format. Please upload WAV or MP3 files."
                )

            file_content = await file.read()
            audio = (
                AudioSegment.from_mp3(io.BytesIO(file_content))
                if file_extension == "mp3"
                else AudioSegment.from_wav(io.BytesIO(file_content))
            )
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000).set_channels(1)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                audio.export(tmp_file.name, format="wav")
                tmp_file_path = tmp_file.name

            chunk_file_paths = asr_manager.split_audio(tmp_file_path)
            try:
                language_id = asr_manager.model_language.get(language, asr_manager.default_language)
                transcription = asr_manager.transcribe(chunk_file_paths, language_id)
                all_transcriptions.append(transcription)
            finally:
                for chunk_file_path in chunk_file_paths:
                    if os.path.exists(chunk_file_path):
                        os.remove(chunk_file_path)
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                asr_manager.cleanup()

        logger.info(f"Batch transcription completed in {time() - start_time:.2f} seconds")
        return JSONResponse(content={"transcriptions": all_transcriptions})
    except Exception as e:
        logger.error(f"Error during batch transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")
'''

@app.post("/v1/translate", response_model=TranslationResponse)
async def translate(
    request: TranslationRequest,
    translate_manager: TranslateManager = Depends(get_translate_manager),
):
    logger.info(f"Received request: {request}")
    try:
        translations = translate_manager.translate(request.sentences)
        return TranslationResponse(translations=translations)
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")



@app.post("/v1/chat_v2", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(...),  # Text prompt as form data
    image: UploadFile = File(default=None),  # Optional image file upload, defaults to None
    api_key: str = Depends(get_api_key)
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {prompt}")

    try:
        decoded = await llm_manager.chat_v2(image, prompt)
        return ChatResponse(response=decoded)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)