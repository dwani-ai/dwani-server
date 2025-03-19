from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from models.tts_manager import TTSModelManager
from tts_config import SPEED, ResponseFormat, config
from utils.helpers import chunk_text
from logging_config import logger
from typing import Annotated, List
import io
import zipfile
import soundfile as sf
import numpy as np
from time import perf_counter
import torch
router = APIRouter()
tts_model_manager = TTSModelManager()

@router.post("/audio/speech")
async def generate_audio(
    input: Annotated[str, Body()] = config.input,
    voice: Annotated[str, Body()] = config.voice,
    model: Annotated[str, Body()] = config.model,
    response_format: Annotated[ResponseFormat, Body(include_in_schema=False)] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = tts_model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning("Specifying speed isn't supported by this model. Audio will be generated with the default speed")
    start = perf_counter()

    cache_key = f"{input}_{voice}_{response_format}"
    if cache_key in tts_model_manager.audio_cache:
        logger.info("Returning cached audio")
        audio_buffer = io.BytesIO(tts_model_manager.audio_cache[cache_key])
        audio_buffer.seek(0)
        return StreamingResponse(audio_buffer, media_type=f"audio/{response_format}")

    all_chunks = chunk_text(input, chunk_size=10)

    cache_key_voice = f"voice_{voice}"
    if cache_key_voice in tts_model_manager.voice_cache:
        desc_inputs = tts_model_manager.voice_cache[cache_key_voice]
        logger.info("Using cached voice description")
    else:
        desc_inputs = description_tokenizer(voice,
                                          return_tensors="pt",
                                          padding="max_length",
                                          max_length=tts_model_manager.max_length).to("cuda" if torch.cuda.is_available() else "cpu")
        tts_model_manager.voice_cache[cache_key_voice] = desc_inputs

    if len(all_chunks) == 1:
        prompt_inputs = tokenizer(input,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=tts_model_manager.max_length).to("cuda" if torch.cuda.is_available() else "cpu")
        
        generation = tts.generate(
            input_ids=desc_inputs["input_ids"],
            prompt_input_ids=prompt_inputs["input_ids"],
            attention_mask=desc_inputs["attention_mask"],
            prompt_attention_mask=prompt_inputs["attention_mask"]
        ).to(torch.float32)
        audio_arr = generation.cpu().float().numpy().squeeze()
    else:
        all_descriptions = [voice] * len(all_chunks)
        description_inputs = description_tokenizer(all_descriptions,
                                                 return_tensors="pt",
                                                 padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
        prompts = tokenizer(all_chunks,
                          return_tensors="pt",
                          padding=True).to("cuda" if torch.cuda.is_available() else "cpu")

        generation = tts.generate(
            input_ids=description_inputs["input_ids"],
            attention_mask=description_inputs["attention_mask"],
            prompt_input_ids=prompts["input_ids"],
            prompt_attention_mask=prompts["attention_mask"],
            do_sample=False,
            return_dict_in_generate=True,
        )
        
        chunk_audios = []
        for i, audio in enumerate(generation.sequences):
            audio_data = audio[:generation.audios_length[i]].cpu().float().numpy().squeeze()
            chunk_audios.append(audio_data)
        audio_arr = np.concatenate(chunk_audios)

    logger.info(f"Took {perf_counter() - start:.2f} seconds to generate audio for {len(input.split())} words")

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_arr, tts.config.sampling_rate, format=response_format)
    audio_buffer.seek(0)
    tts_model_manager.audio_cache[cache_key] = audio_buffer.getvalue()
    return StreamingResponse(audio_buffer, media_type=f"audio/{response_format}")

@router.post("/audio/speech_batch")
async def generate_audio_batch(
    input: Annotated[List[str], Body()] = config.input,
    voice: Annotated[List[str], Body()] = config.voice,
    model: Annotated[str, Body(include_in_schema=False)] = config.model,
    response_format: Annotated[ResponseFormat, Body()] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = tts_model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning("Specifying speed isn't supported by this model. Audio will be generated with the default speed")
    start = perf_counter()

    cached_outputs = []
    uncached_inputs = []
    uncached_voices = []
    cache_keys = [f"{text}_{voice[i]}_{response_format}" for i, text in enumerate(input)]
    for i, key in enumerate(cache_keys):
        if key in tts_model_manager.audio_cache:
            cached_outputs.append((i, tts_model_manager.audio_cache[key]))
        else:
            uncached_inputs.append(input[i])
            uncached_voices.append(voice[i])

    if uncached_inputs:
        all_chunks = []
        all_descriptions = []
        for i, text in enumerate(uncached_inputs):
            chunks = chunk_text(text, chunk_size=10)
            all_chunks.extend(chunks)
            all_descriptions.extend([uncached_voices[i]] * len(chunks))

        unique_descriptions = list(set(all_descriptions))
        desc_inputs_dict = {}
        for desc in unique_descriptions:
            cache_key_voice = f"voice_{desc}"
            if cache_key_voice in tts_model_manager.voice_cache:
                desc_inputs_dict[desc] = tts_model_manager.voice_cache[cache_key_voice]
            else:
                desc_inputs = description_tokenizer(desc,
                                                  return_tensors="pt",
                                                  padding="max_length",
                                                  max_length=tts_model_manager.max_length).to("cuda" if torch.cuda.is_available() else "cpu")
                desc_inputs_dict[desc] = desc_inputs
                tts_model_manager.voice_cache[cache_key_voice] = desc_inputs

        description_inputs = description_tokenizer(all_descriptions,
                                                 return_tensors="pt",
                                                 padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
        prompts = tokenizer(all_chunks,
                          return_tensors="pt",
                          padding=True).to("cuda" if torch.cuda.is_available() else "cpu")

        generation = tts.generate(
            input_ids=description_inputs["input_ids"],
            attention_mask=description_inputs["attention_mask"],
            prompt_input_ids=prompts["input_ids"],
            prompt_attention_mask=prompts["attention_mask"],
            do_sample=False,
            return_dict_in_generate=True,
        )

        audio_outputs = []
        current_index = 0
        for i, text in enumerate(uncached_inputs):
            chunks = chunk_text(text, chunk_size=10)
            chunk_audios = []
            for _ in range(len(chunks)):
                audio_arr = generation.sequences[current_index][:generation.audios_length[current_index]].cpu().float().numpy().squeeze()
                chunk_audios.append(audio_arr)
                current_index += 1
            combined_audio = np.concatenate(chunk_audios)
            audio_outputs.append(combined_audio)

        for i, (text, voice_) in enumerate(zip(uncached_inputs, uncached_voices)):
            key = f"{text}_{voice_}_{response_format}"
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_outputs[i], tts.config.sampling_rate, format=response_format)
            audio_buffer.seek(0)
            tts_model_manager.audio_cache[key] = audio_buffer.getvalue()

    final_outputs = [None] * len(input)
    for idx, audio_data in cached_outputs:
        final_outputs[idx] = audio_data
    uncached_idx = 0
    for i in range(len(final_outputs)):
        if final_outputs[i] is None:
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_outputs[uncached_idx], tts.config.sampling_rate, format=response_format)
            audio_buffer.seek(0)
            final_outputs[i] = audio_buffer.getvalue()
            uncached_idx += 1

    file_data = {f"out_{i}.{response_format}": data for i, data in enumerate(final_outputs)}
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, 'w') as zipf:
        for file_name, data in file_data.items():
            zipf.writestr(file_name, data)
    in_memory_zip.seek(0)

    logger.info(f"Took {perf_counter() - start:.2f} seconds to generate audio batch")
    return StreamingResponse(in_memory_zip, media_type="application/zip")