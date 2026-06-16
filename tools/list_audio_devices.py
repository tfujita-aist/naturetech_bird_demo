# list_audio_devices.py
import sounddevice as sd
print(sd.query_devices())
