#!/usr/bin/env python3
"""CosyVoice3 NPU inference verification."""

import os
import sys
import time
import json

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "6,7"

import torch
import torch_npu

print(f"NPU available: {torch.npu.is_available()}, count: {torch.npu.device_count()}")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "third_party/Matcha-TTS"))
sys.path.insert(0, ROOT_DIR)

import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging

MODEL_DIR = os.path.expanduser(
    "~/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512"
)
PROMPT_WAV = os.path.join(ROOT_DIR, "asset/zero_shot_prompt.wav")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output_npu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS = {}


def log_result(name, elapsed, wav_path, speech):
    dur = speech.shape[1] / 24000
    print(f"[CosyVoice3-NPU] {name} ({elapsed:.2f}s, dur={dur:.2f}s) -> {wav_path}")
    RESULTS[name] = {"time": round(elapsed, 2), "duration": round(dur, 2), "output": wav_path}


def main():
    print("CosyVoice3 NPU Inference Verification")
    print("=" * 60)

    cosyvoice = AutoModel(model_dir=MODEL_DIR, fp16=True)
    print(f"Device: {cosyvoice.model.device}")
    print(f"Sample rate: {cosyvoice.sample_rate}")

    prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

    print("\n--- Test 1: zero-shot (中文) ---")
    t0 = time.time()
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
        prompt_text, PROMPT_WAV, stream=False
    )):
        out = os.path.join(OUTPUT_DIR, f"zero_shot_{i}.wav")
        torchaudio.save(out, j["tts_speech"], cosyvoice.sample_rate)
        log_result(f"zero_shot_{i}", time.time() - t0, out, j["tts_speech"])

    print("\n--- Test 2: zero-shot (English) ---")
    t0 = time.time()
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        prompt_text, PROMPT_WAV, stream=False
    )):
        out = os.path.join(OUTPUT_DIR, f"zero_shot_en_{i}.wav")
        torchaudio.save(out, j["tts_speech"], cosyvoice.sample_rate)
        log_result(f"zero_shot_en_{i}", time.time() - t0, out, j["tts_speech"])

    print("\n--- Test 3: cross-lingual ---")
    t0 = time.time()
    for i, j in enumerate(cosyvoice.inference_cross_lingual(
        "You are a helpful assistant.<|endofprompt|>因为他们那一辈人在乡里面住的要习惯一点。",
        PROMPT_WAV, stream=False
    )):
        out = os.path.join(OUTPUT_DIR, f"cross_lingual_{i}.wav")
        torchaudio.save(out, j["tts_speech"], cosyvoice.sample_rate)
        log_result(f"cross_lingual_{i}", time.time() - t0, out, j["tts_speech"])

    print("\n--- Test 4: instruct ---")
    t0 = time.time()
    for i, j in enumerate(cosyvoice.inference_instruct2(
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
        PROMPT_WAV, stream=False
    )):
        out = os.path.join(OUTPUT_DIR, f"instruct_{i}.wav")
        torchaudio.save(out, j["tts_speech"], cosyvoice.sample_rate)
        log_result(f"instruct_{i}", time.time() - t0, out, j["tts_speech"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    for k, v in RESULTS.items():
        print(f"  {k}: {v['time']}s (dur={v['duration']}s) -> {v['output']}")

    with open(os.path.join(OUTPUT_DIR, "npu_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print("\nResults saved to npu_results.json")


if __name__ == "__main__":
    main()
