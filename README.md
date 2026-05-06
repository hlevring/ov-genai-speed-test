# ov-genai-speed-test

Benchmark Whisper transcription speed using OpenVINO GenAI.

## Setup

Clone the repository:

```bash
git clone https://github.com/hlevring/ov-genai-speed-test.git
cd ov-genai-speed-test
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the benchmark:

```bash
python -m ov_asr_speed -m hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d cpu
```

Run it a second time for a warm-cache comparison — the model is downloaded on
the first run and cached by HuggingFace, so the second run reflects
steady-state performance.

### Arguments

| Argument | Short | Required | Description |
|---|---|---|---|
| `--model` | `-m` | yes | Local model directory **or** HuggingFace repo ID |
| `--audio` | `-a` | yes | Path to a WAV audio file |
| `--device` | `-d` | yes | `cpu`, `gpu`, or `npu` |
| `--cache_dir` | | no | OpenVINO compilation cache directory |

### Notes

- GPU and NPU require an Intel device; CPU accepts any vendor.
- NPU automatically enables `STATIC_PIPELINE`.
- If `--model` is not a local directory, the script downloads it via `huggingface_hub.snapshot_download()`.
- `--cache_dir` with CPU is expected to crash due to [openvinotoolkit/openvino#35379](https://github.com/openvinotoolkit/openvino/issues/35379). Use it only with GPU or NPU.
