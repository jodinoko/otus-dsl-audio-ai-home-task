import time
from pathlib import Path

import soundfile as sf
import torch

SAMPLE_RATE = 48000
SPEAKER = "kseniya"
OUTPUT_DIR = Path("outputs/audio")

_model = None


def load_silero():
    global _model
    if _model is None:
        _model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v5_5_ru",
            trust_repo=True,
        )
    return _model


def synthesize(text):
    audio = load_silero().apply_tts(
        text=text,
        speaker=SPEAKER,
        sample_rate=SAMPLE_RATE,
        put_accent=True,
        put_yo=True,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"answer_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    sf.write(path, audio.numpy(), SAMPLE_RATE)
    return str(path)
