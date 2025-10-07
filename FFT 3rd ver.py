import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import soundfile as sf

def format_time(x, pos):        # Function: format time axis as 70s -> 1:10
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{minutes}:{seconds:02d}"

def format_k(x, pos):        # Function: format frequency axis as 2000 -> 2k
    return f"{x/1000:.0f}k" if x >= 1000 or x == 0 else f"{int(x)}"

# === 1. Load audio ===
filename = "Dewa - Dua Sedjoli.mp3"   # <<<< input file here
data, fs = sf.read(filename, dtype='float32')

# If stereo, convert to mono
if data.ndim > 1:
    data = data.mean(axis=1)

# === Normalize ===
data /= np.max(np.abs(data))

# === 2. Time & duration ===
N = len(data)
duration = N / fs
time = np.linspace(0, duration, N)

# === 3. FFT ===
fft_data = np.fft.fft(data)
freqs = np.fft.fftfreq(N, 1/fs)

# === Set figure size ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# === 4. Plot waveform ===
ax1.plot(time, data, linewidth=0.8)
ax1.set_title("Waveform")
ax1.set_xlabel("Time (m:ss)")
ax1.set_ylabel("Amplitude")

# -- Margins for waveform --
margin = duration * 0.01
ax1.set_xlim(-margin, duration + margin)

# -- Ticks for waveform --
tick_interval = 1 if duration <= 10 else int(duration // 10)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
ax1.grid(True, linestyle='--', alpha=0.3)

# === 5. Plot frequency ===
ax2.plot(freqs[:N//2], np.abs(fft_data[:N//2]), linewidth=0.8)
ax2.set_title("FFT Frequency Spectrum")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude", labelpad=9)

# -- Margins for frequency --
freq_end = fs / 2
freq_margin = freq_end * 0.01
ax2.set_xlim(-freq_margin, freq_end + freq_margin)

# -- Ticks for frequency --
freq_tick = 1000 if fs <= 48000 else fs // 40
ax2.xaxis.set_major_locator(ticker.MultipleLocator(freq_tick))
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_k))
ax2.grid(True, linestyle='--', alpha=0.3)

# Additional magnitude format
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_k))

# === Set window title ===
plt.gcf().canvas.manager.set_window_title("FFT 3rd ver.py")
plt.tight_layout()
plt.show()
