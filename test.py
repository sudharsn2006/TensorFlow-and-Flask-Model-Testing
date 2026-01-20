import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Set female voice (Zira)
voices = engine.getProperty('voices')
for voice in voices:
    if "Zira" in voice.name:
        engine.setProperty('voice', voice.id)
        break

# Optional: Adjust speaking speed
engine.setProperty('rate', 150)

# Test speaking
engine.say("Hello, I am your image recognition assistant")
engine.runAndWait()
