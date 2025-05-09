import dwani

dwani.api_key = "your_key"
result = dwani.chat.create(prompt="Hello!")
audio = dwani.audio.speech(input="Hello", voice="Female", model="tts-model")
