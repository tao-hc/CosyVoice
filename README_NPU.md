# CosyVoice3 昇腾 NPU 快速入门

## 环境准备

```bash
conda activate torch280_py310_ali

# 关键：transformers 必须使用 4.51.3，高版本会导致音频失真
pip install transformers==4.51.3 --no-deps
pip install "tokenizers>=0.21,<0.22" --no-deps
pip install "huggingface-hub>=0.30.0,<1.0" --no-deps
```

## 下载模型权重

```bash
unset https_proxy http_proxy ALL_PROXY
python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512')"
```

## 运行推理

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7
cd 01_CosyVoice
python run_npu.py
```

输出音频保存在 `output_npu/` 目录下。

## 已验证功能

- [x] zero-shot TTS（中文 + 英文）
- [x] cross-lingual TTS
- [x] instruct TTS
- [x] fp16 autocast 推理

## 注意事项

1. **transformers 版本**: 必须使用 4.51.3，5.x 版本会导致 Qwen2 LLM 生成的 speech tokens 失真
2. **iSTFT**: 用 `torch.fft.irfft` + `scatter_add` 替代 `torch.istft`（NPU 不支持 fold 算子），全程在 NPU 上执行
3. **float64**: CausalHiFTGenerator 的 f0_predictor 需要 float64，仅此模块在 CPU 上执行
4. **推理性能**: fp16 模式下 RTF ≈ 1.5-1.8（除首次含编译开销外）
