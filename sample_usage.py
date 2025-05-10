import dwani
import os

dwani.api_key = os.getenv("DWANI_API_KEY")

dwani.api_base = os.getenv("DWANI_API_BASE")

resp = dwani.Chat.create("Hello!", "eng_Latn", "kan_Knda")
print(resp)




'''
# TTS
dwani.audio.speech("Hello world", output_file="speech.mp3")

# ASR
trans = dwani.asr.transcribe("audio.wav", "kannada")
print(trans)

# Translate
trans = dwani.translate.text(["Hello", "How are you?"], "eng_Latn", "kan_Knda")
print(trans)





import dwani

dwani.api_key = "your_api_key"

# OCR
ocr_result = dwani.documents.ocr("page1.png", language="eng_Latn")
print(ocr_result)

# Translate
trans_result = dwani.documents.translate("letter.pdf", src_lang="eng_Latn", tgt_lang="hin_Deva")
print(trans_result)

# Summarize
summary = dwani.documents.summarize("report.pdf")
print(summary)
'''