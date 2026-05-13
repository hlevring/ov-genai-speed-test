# ov-genai-speed-test

Benchmark Whisper transcription speed using OpenVINO GenAI.

## Prerequisites

- **Python 3.10+** 

## Setup

Clone the repository:

```bash
git clone https://github.com/hlevring/ov-genai-speed-test.git
cd ov-genai-speed-test
```

Create a virtual environment:

```bash
# Windows
python -m venv .venv

# Linux / macOS
python3 -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies (latest stable release):

```bash
pip install -r requirements.txt
```

Or install with nightly OpenVINO builds:

```bash
pip install -r requirements_nightly.txt
```

## Usage

Make sure the virtual environment is activated, then run the benchmark:

Whisper Small Int8:

```bash
python -m ov_asr_speed -m hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d cpu
```

Whisper Large V3 Turbo Int8:

```bash
python -m ov_asr_speed -m hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 --audio demoB108s.wav -d cpu
```

Run it a second time for a warm-cache comparison — the model is downloaded on
the first run and cached by HuggingFace, so the second run reflects
steady-state performance.

### Arguments

| Argument | Short | Required | Description |
|---|---|---|---|
| `--model` | `-m` | yes | Local model directory **or** HuggingFace repo ID |
| `--audio` | `-a` | yes | Path to a WAV audio file |
| `--device` | `-d` | yes | `CPU`, `GPU`, `GPU.0`, `GPU.1`, `NPU`, etc. (`GPU` resolves to first Intel GPU) |
| `--cache_dir` | | no | OpenVINO compilation cache directory |
| `--static` | | no | Use `STATIC_PIPELINE` (required for openvino-genai <= 2026.1, not required with latest nightly) |
| `--weight-less-caching` | `-wl` | no | Set `CACHE_MODE=OPTIMIZE_SIZE` for compilation cache (plugin-dependent support) |

### Notes

- `GPU` is treated as an Intel-only alias and resolves to the first Intel GPU device (for example `GPU.1`).
- `GPU.0` / `GPU.1` remain supported, but must refer to an Intel GPU.
- If no Intel GPU is available for `GPU`, the script exits with a clear error.
- NPU requires an Intel device; CPU accepts any vendor.
- If `--model` is not a local directory, the script downloads it via `huggingface_hub.snapshot_download()`.
- `--cache_dir` with CPU is expected to crash due to [openvinotoolkit/openvino#35379](https://github.com/openvinotoolkit/openvino/issues/35379). Use it only with GPU or NPU.
- `--weight-less-caching/-wl` requires `--cache_dir`.
- `--weight-less-caching/-wl` is accepted for all devices, but whether `CACHE_MODE` is supported depends on the active plugin/build.
