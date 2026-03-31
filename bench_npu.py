#!/usr/bin/env python3
"""CosyVoice3 NPU performance benchmark: 3 warmup + 1 measured run."""
import os, sys, time, json

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "6,7"

import torch
import torch_npu

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "third_party/Matcha-TTS"))

import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

MODEL_DIR = os.path.expanduser("~/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512")
PROMPT_WAV = os.path.join(ROOT, "asset/zero_shot_prompt.wav")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
OUT = os.path.join(ROOT, "output_npu")
os.makedirs(OUT, exist_ok=True)

cosyvoice = AutoModel(model_dir=MODEL_DIR, fp16=True)
print(f"Device: {cosyvoice.model.device}, fp16: {cosyvoice.model.fp16}")

tests = [
    ("zero_shot_zh", "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
     lambda text: cosyvoice.inference_zero_shot(text, PROMPT_TEXT, PROMPT_WAV, stream=False)),
    ("zero_shot_en", "The quick brown fox jumps over the lazy dog near the river bank.",
     lambda text: cosyvoice.inference_zero_shot(text, PROMPT_TEXT, PROMPT_WAV, stream=False)),
    ("cross_lingual", "You are a helpful assistant.<|endofprompt|>因为他们那一辈人在乡里面住的要习惯一点。",
     lambda text: cosyvoice.inference_cross_lingual(text, PROMPT_WAV, stream=False)),
    ("instruct", "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
     lambda text: cosyvoice.inference_instruct2(text,
         "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>", PROMPT_WAV, stream=False)),
]

WARMUP = 3
results = {}

for name, text, infer_fn in tests:
    print(f"\n{'='*60}")
    print(f"=== {name} ===")

    for run_idx in range(WARMUP + 1):
        is_measured = (run_idx == WARMUP)
        label = f"run {run_idx+1}" if not is_measured else "MEASURED"
        set_all_random_seed(0)

        torch.npu.synchronize()
        t0 = time.time()
        for i, j in enumerate(infer_fn(text)):
            speech = j["tts_speech"]
        torch.npu.synchronize()
        elapsed = time.time() - t0

        dur = speech.shape[1] / cosyvoice.sample_rate
        rtf = elapsed / dur
        print(f"  [{label}] time={elapsed:.2f}s dur={dur:.2f}s rtf={rtf:.3f}")

        if is_measured:
            out_path = os.path.join(OUT, f"{name}.wav")
            torchaudio.save(out_path, speech, cosyvoice.sample_rate)
            results[name] = {
                "time_s": round(elapsed, 2),
                "duration_s": round(dur, 2),
                "rtf": round(rtf, 3),
            }

print(f"\n{'='*60}")
print("BENCHMARK RESULTS (after 3 warmup):")
print(f"{'Test':<20} {'Time(s)':<10} {'Duration(s)':<12} {'RTF':<8}")
print("-" * 50)
for k, v in results.items():
    print(f"{k:<20} {v['time_s']:<10} {v['duration_s']:<12} {v['rtf']:<8}")

with open(os.path.join(OUT, "benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {OUT}/benchmark_results.json")
