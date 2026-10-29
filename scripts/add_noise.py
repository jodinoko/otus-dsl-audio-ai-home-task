from pathlib import Path

import numpy as np
import soundfile as sf

SNR_LEVELS = {"strong": 0, "very_strong": -5}
SOURCE_DIR = Path("data/audio")
OUTPUT_DIR = Path("data/audio_noisy")


def add_noise(audio, snr_db, rng):
    noise = rng.normal(size=audio.shape).astype("float32")
    signal_power = np.mean(audio**2)
    noise_power = signal_power / 10 ** (snr_db / 10)
    noisy = audio + noise * np.sqrt(noise_power)
    return noisy / max(1.0, np.abs(noisy).max())


def main():
    rng = np.random.default_rng(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(SOURCE_DIR.glob("*.mp3")):
        audio, sr = sf.read(path, dtype="float32")
        for level, snr_db in SNR_LEVELS.items():
            output = OUTPUT_DIR / f"{path.stem}_{level}.wav"
            sf.write(output, add_noise(audio, snr_db, rng), sr)
            print(f"{output} (SNR {snr_db} дБ)")


if __name__ == "__main__":
    main()
