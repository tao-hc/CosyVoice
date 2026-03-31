# CosyVoice3 昇腾 NPU 适配报告

## 基本信息

| 项目 | 值 |
|------|------|
| 模型 | Fun-CosyVoice3-0.5B-2512 (ModelScope: FunAudioLLM/Fun-CosyVoice3-0___5B-2512) |
| 框架 | CosyVoice (FunAudioLLM) |
| NPU 卡 | ASCEND_RT_VISIBLE_DEVICES=6,7 |
| Conda 环境 | torch280_py310_ali |
| PyTorch | 2.8.0 + torch_npu |
| transformers | **4.51.3**（关键） |
| 采样率 | 24000 Hz |
| 推理模式 | fp16 autocast (LLM + Flow) |

## 代码改动

### 1. `cosyvoice/cli/model.py` — 设备适配

- 新增 `_get_device()`: 自动检测 CUDA → NPU → CPU
- 新增 `_make_stream_context()`: 为 NPU 创建 `torch.npu.stream` 上下文
- 新增 `_empty_cache()`: NPU 缓存清理
- CosyVoiceModel / CosyVoice2Model / CosyVoice3Model:
  - `self.device = _get_device()` 替代硬编码 `torch.device('cuda')`
  - `self.llm_context = _make_stream_context(self.device)` 替代 `torch.cuda.stream(...)`
  - `torch.amp.autocast(device_type=..., enabled=...)` 替代 `torch.cuda.amp.autocast(...)`
  - `_empty_cache(self.device)` 替代 `torch.cuda.empty_cache()`
  - `tts_mel = tts_mel.float()` 确保 HiFT 输入为 float32

### 2. `cosyvoice/cli/cosyvoice.py` — 加速器检测

- 3 处 `torch.cuda.is_available()` 改为 `_has_accelerator = torch.cuda.is_available() or (hasattr(torch, 'npu') and torch.npu.is_available())`
- 防止 NPU 环境下错误禁用 fp16 / load_jit 等

### 3. `cosyvoice/cli/frontend.py` — 设备与 ONNX Runtime

- 设备选择增加 NPU 分支
- ONNX Runtime `speech_tokenizer_session` 在 NPU 上回退 `CPUExecutionProvider`

### 4. `cosyvoice/utils/common.py` — 随机种子

- `set_all_random_seed()` 增加 `torch.npu.manual_seed_all(seed)`

### 5. `cosyvoice/hifigan/generator.py` — iSTFT 纯 NPU 实现 + HiFT 全 NPU

- `HiFTGenerator._istft()`: 用 `torch.fft.irfft` + `scatter_add` 重写 overlap-add，替代 `torch.istft`（因 NPU 不支持 `aclnnUnfoldGrad`/`fold` 算子）
- `HiFTGenerator._stft()`: 保持 `torch.stft`（NPU 支持前向 `unfold`）
- `HiFTGenerator.inference()`: 移除整体 `self.cpu()` 搬迁，全程在 NPU 执行
- `CausalHiFTGenerator.inference()`: f0_predictor 改为 fp32（原 fp64，实测精度无损），全程 NPU 执行，无 CPU 回退

## 关键问题记录

### transformers 版本导致音频质量严重失真

| transformers 版本 | 音频质量 | 音频 std | 现象 |
|------|------|------|------|
| **5.3.0** | ❌ 严重失真 | 0.0431 | "ororor" 声，听不清内容，语调异常 |
| **4.51.3** | ✅ 正常 | 0.1082 | 发音清晰，语调自然 |

**根因**: `transformers>=5.0` 对 Qwen2 模型的内部 attention 计算有行为变更，导致 LLM 生成的 speech tokens 质量下降。该问题**在 CPU 和 NPU 上均可复现**，与硬件无关。

**修复**: 锁定 `transformers==4.51.3`，配套 `tokenizers>=0.21,<0.22` 和 `huggingface-hub>=0.30.0,<1.0`。

### f0_predictor 精度

原代码要求 `CausalHiFTGenerator` 的 `f0_predictor` 使用 float64。实测 fp32 精度无损（音频 std 差异 < 0.0002），已改为 fp32 在 NPU 上直接运行，无需 CPU 回退。

### NPU 不支持 torch.istft

`torch.istft` 内部使用 `fold` 算子（底层 `aclnnUnfoldGrad`），当前 CANN 版本不支持。已用 `torch.fft.irfft` + `scatter_add` 手动实现 overlap-add，完全在 NPU 上执行，无需 CPU 回退。

## 端到端验证结果 (fp16 + 全 NPU，3次 warmup 后)

| 测试项 | 推理时间 | 音频时长 | RTF |
|------|------|------|------|
| zero_shot (中文) | 15.65s | 9.04s | 1.731 |
| zero_shot (English) | 8.42s | 4.52s | 1.862 |
| cross_lingual | 8.07s | 4.16s | 1.941 |
| instruct | 9.06s | 5.16s | 1.755 |

3 次 warmup 后平均 RTF ≈ **1.82**。

## 优化效果

| 优化阶段 | 代表推理时间 (中文 zero-shot) | RTF | 加速比 |
|------|------|------|------|
| 基线 (fp32 + HiFT CPU 回退) | 41.72s | 4.31 | 1.00x |
| + fp16 autocast | 37.65s | 4.16 | 1.04x |
| + iSTFT NPU + f0 fp32 + 去 CPU 回退 | 15.65s (warmup 后) | 1.73 | **2.49x** |

- fp16 autocast 对 LLM（Qwen2）和 Flow（DiT）生效
- 自定义 iSTFT（`irfft` + `scatter_add`）替代 `torch.istft`，避免 NPU 不支持的 `fold` 算子
- f0_predictor 从 fp64@CPU 改为 fp32@NPU（实测精度无损）
- `_empty_cache` NPU 部分默认关闭，通过 `COSYVOICE_NPU_EMPTY_CACHE=1` 环境变量开启，减少同步开销

## 依赖版本锁定

```
transformers==4.51.3
tokenizers>=0.21,<0.22
huggingface-hub>=0.30.0,<1.0
# torch / torch_npu 保持现有版本不变
```

## 运行方式

```bash
conda activate torch280_py310_ali
export ASCEND_RT_VISIBLE_DEVICES=6,7
unset https_proxy http_proxy ALL_PROXY
cd 01_CosyVoice
python run_npu.py
```
