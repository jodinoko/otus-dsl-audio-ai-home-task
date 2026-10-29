import time

from audio_ai.asr import transcribe
from audio_ai.tts import synthesize
from audio_ai.vlm import ask


def run_omni_assistant(image, audio_query):
    latency = {}

    start = time.perf_counter()
    question = transcribe(audio_query)
    latency["asr"] = time.perf_counter() - start

    start = time.perf_counter()
    answer = ask(image, question)
    latency["vlm"] = time.perf_counter() - start

    start = time.perf_counter()
    audio_path = synthesize(answer)
    latency["tts"] = time.perf_counter() - start

    latency["total"] = sum(latency.values())
    print(f"Вопрос: {question}")
    print(f"Ответ: {answer}")
    print(" | ".join(f"{stage}: {seconds:.1f} c" for stage, seconds in latency.items()))
    return answer, audio_path, question, latency
