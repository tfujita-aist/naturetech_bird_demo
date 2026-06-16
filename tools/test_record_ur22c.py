# test_record_ur22c.py
import sounddevice as sd
import soundfile as sf

DEVICE = 7
SR = 48000
SECONDS = 5
CHANNELS = 2

print(sd.query_devices())

print("Recording 5 seconds from Steinberg UR22C...")
x = sd.rec(
    int(SR * SECONDS),
    samplerate=SR,
    channels=CHANNELS,
    dtype="float32",
    device=DEVICE,
)
sd.wait()

sf.write("test_input1_umg15b.wav", x[:, 0], SR, subtype="PCM_16")
sf.write("test_input2_um230c.wav", x[:, 1], SR, subtype="PCM_16")

print("Saved:")
print("  test_input1_umg15b.wav")
print("  test_input2_um230c.wav")