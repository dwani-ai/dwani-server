from dwani import Chat

resp = Chat.create("Hello!", "eng_Latn", "kan_Knda")
print(resp)

'''


import dwani

dwani.api_key = "your_api_key"
base_url = "https://dwani-dwani-server-workshop.hf.space"

dwani.api_base = "http://localhost:7860"

dwani.api_base = base_url
# Chat
resp = dwani.chat.create("Hello!", "eng_Latn", "kan_Knda")

print(resp)
'''


'''
from dwani import chat

# Set API key if needed

dwani.api_key = "your_api_key"
base_url = "https://dwani-dwani-server-workshop.hf.space"
dwani.api_base = base_url

resp = chat.create("Hello!", lang_from="eng_Latn", lang_to="kan_Knda")
print(resp)

'''

'''
import dwani

dwani.api_key = "your_api_key"
base_url = "https://dwani-dwani-server-workshop.hf.space"

dwani.api_base = "http://localhost:7860"

dwani.api_base = base_url
# Chat
#resp = dwani.chat.create("Hello!", "eng_Latn", "kan_Knda")
resp = dwani.chat.create("Hello!") #, "eng_Latn", "kan_Knda")

print(resp)
'''
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