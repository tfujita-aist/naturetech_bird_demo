import sys
import numpy as np
import soundfile as sf

path = sys.argv[1]

y, sr = sf.read(path, always_2d=True)
print("path:", path)
print("sample_rate:", sr)
print("shape:", y.shape)

for ch in range(y.shape[1]):
    x = y[:, ch]
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    print(f"CH{ch+1}: peak={peak:.6f}, rms={rms:.6f}")