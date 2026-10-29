import noisereduce as nr
import soundfile as sf
import torch
import torchaudio
import whisper

SAMPLE_RATE = 16000

_model = None


def load_whisper(name="small"):
    global _model
    if _model is None:
        _model = whisper.load_model(name, device="cpu")
    return _model


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(axis=1))
    return torchaudio.functional.resample(audio, sr, SAMPLE_RATE).numpy()


def transcribe(path):
    audio = load_audio(path)
    audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)
    result = load_whisper().transcribe(audio, language="ru", fp16=False)
    return result["text"].strip()
