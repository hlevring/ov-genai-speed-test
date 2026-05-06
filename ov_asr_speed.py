"""Benchmark Whisper transcription speed with OpenVINO GenAI."""

import argparse
import os
import platform
import re
import sys
import time

import numpy as np


def resolve_model(model: str) -> str:
    """Return a local directory path, downloading from HuggingFace if needed."""
    if os.path.isdir(model):
        return model

    from huggingface_hub import snapshot_download

    print(f"Model not found locally — downloading {model} from HuggingFace …")
    return snapshot_download(repo_id=model)


def sanitize_filename(name: str) -> str:
    """Turn a device full name into a safe filename stem."""
    name = re.sub(r"\(R\)|\(TM\)|\(r\)|\(tm\)", "", name)
    name = re.sub(r"[^\w\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def get_device_name(device: str) -> str:
    """Validate device and return its full name from OpenVINO."""
    import openvino as ov

    core = ov.Core()

    if device != "CPU":
        available = core.available_devices
        if device not in available:
            sys.exit(
                f"Device {device} is not available. "
                f"Available devices: {', '.join(available)}"
            )
        full_name = core.get_property(device, "FULL_DEVICE_NAME")
        if "intel" not in full_name.lower():
            sys.exit(f"Device {device} is not Intel: {full_name}")
    else:
        full_name = core.get_property("CPU", "FULL_DEVICE_NAME")

    print(f"Device {device}: {full_name}")
    return full_name


def load_audio(path: str) -> tuple[np.ndarray, float]:
    """Load a WAV file as 16 kHz float32 mono and return (samples, duration_s)."""
    import librosa

    samples, _ = librosa.load(path, sr=16000, mono=True)
    duration_s = len(samples) / 16000
    return samples, duration_s


def build_pipeline(model_path: str, device: str, cache_dir: str | None):
    """Construct the WhisperPipeline and return (pipe, load_ms)."""
    import openvino_genai as ov_genai

    props: dict = {}
    if cache_dir:
        if device == "CPU":
            print("WARNING: CACHE_DIR on CPU may crash — "
                  "see https://github.com/openvinotoolkit/openvino/issues/35379")
        props["CACHE_DIR"] = cache_dir
    if device == "NPU":
        props["STATIC_PIPELINE"] = True

    t0 = time.perf_counter()
    pipe = ov_genai.WhisperPipeline(model_path, device, **props)
    load_ms = (time.perf_counter() - t0) * 1000

    return pipe, load_ms


def run_inference(pipe, samples: np.ndarray):
    """Run timed transcription and return (result, inference_ms)."""
    config = pipe.get_generation_config()
    config.task = "transcribe"
    config.return_timestamps = True
    config.max_new_tokens = 256

    t0 = time.perf_counter()
    result = pipe.generate(samples.tolist(), config)
    inference_ms = (time.perf_counter() - t0) * 1000

    return result, inference_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper transcription speed with OpenVINO GenAI",
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Local model directory or HuggingFace repo ID "
             "(e.g. hlevring/ov-whisper_small-int8-2026.0.0)",
    )
    parser.add_argument(
        "--audio", "-a", required=True,
        help="Path to a WAV audio file",
    )
    parser.add_argument(
        "--device", "-d", required=True,
        choices=["cpu", "gpu", "npu", "CPU", "GPU", "NPU"],
        help="Inference device: cpu, gpu, or npu",
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="OpenVINO compilation cache directory",
    )
    args = parser.parse_args()

    device = args.device.upper()

    # --- Validate device --------------------------------------------------
    device_full_name = get_device_name(device)

    # --- Resolve model path -----------------------------------------------
    model_path = resolve_model(args.model)
    model_label = args.model

    # --- Load audio --------------------------------------------------------
    if not os.path.isfile(args.audio):
        sys.exit(f"Audio file not found: {args.audio}")

    print(f"Loading audio: {args.audio}")
    t0 = time.perf_counter()
    samples, duration_s = load_audio(args.audio)
    audio_load_ms = (time.perf_counter() - t0) * 1000
    print(f"Audio loaded: {duration_s:.1f}s ({audio_load_ms:.0f} ms)")

    # --- Build pipeline (load + compile) -----------------------------------
    print(f"Loading model on {device} …")
    pipe, load_ms = build_pipeline(model_path, device, args.cache_dir)
    print(f"Model loaded in {load_ms:.0f} ms")

    # --- Timed inference ---------------------------------------------------
    print("Running timed inference …")
    result, inference_ms = run_inference(pipe, samples)

    # --- Summary -----------------------------------------------------------
    realtime_factor = duration_s / (inference_ms / 1000)
    total_ms = load_ms + inference_ms

    text = ""
    if hasattr(result, "texts") and result.texts:
        text = result.texts[0].strip()
    elif hasattr(result, "text"):
        text = result.text.strip() if result.text else ""

    cmdline = " ".join(sys.argv)
    preview = text[:200] + ("…" if len(text) > 200 else "")

    lines = [
        "=" * 60,
        f"{'Command:':<18}{cmdline}",
        f"{'Platform:':<18}{platform.platform()}",
        f"{'Device:':<18}{device} ({device_full_name})",
        f"{'Model:':<18}{model_label}",
        f"{'Audio:':<18}{os.path.basename(args.audio)} ({duration_s:.1f}s)",
        "",
        f"{'Load+compile:':<18}{load_ms:>8.0f} ms",
        f"{'Inference:':<18}{inference_ms:>8.0f} ms  ({realtime_factor:.1f}x realtime)",
        f"{'Total:':<18}{total_ms:>8.0f} ms",
        "",
        "Transcript (first 200 chars):",
        f'  "{preview}"',
        "=" * 60,
    ]
    report = "\n".join(lines)

    print()
    print(report)

    # --- Save to file ------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    stem = f"{device}_{sanitize_filename(device_full_name)}"
    result_path = os.path.join(results_dir, f"{stem}.txt")
    counter = 2
    while os.path.exists(result_path):
        result_path = os.path.join(results_dir, f"{stem}_{counter}.txt")
        counter += 1

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\nResult saved to {result_path}")


if __name__ == "__main__":
    main()
