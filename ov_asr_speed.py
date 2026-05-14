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


def short_device_name(full_name: str) -> str:
    """Extract a compact processor identifier from the OpenVINO FULL_DEVICE_NAME."""
    name = re.sub(r"\(R\)|\(TM\)|\(r\)|\(tm\)", "", full_name)
    name = re.sub(r"\b(Intel|Processor)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"@.*", "", name)
    name = re.sub(r"\b\d+th Gen\b", "", name)
    name = name.strip()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


def short_model_name(model_arg: str) -> str:
    """Extract a compact Whisper variant from the model argument string."""
    base = model_arg.rsplit("/", 1)[-1]
    m = re.search(r"whisper[_-](.+?)(?:-int\d|-fp\d|-uint\d|-bnb\d|-q\d)", base,
                  re.IGNORECASE)
    if not m:
        return base
    variant = m.group(1)
    variant = variant.replace("large_v3_turbo", "V3-turbo")
    variant = variant.replace("large_v3", "V3")
    variant = variant.replace("large_v2", "V2")
    variant = variant.replace("large", "large")
    variant = re.sub(r"[_]+", "-", variant)
    return variant


def result_filename_stem(device: str, device_full_name: str,
                         model_arg: str) -> str:
    """Build a short but distinctive filename stem like NPU_Core-Ultra-7-255H_V3-turbo."""
    prefix = device.split(".")[0]
    return f"{prefix}_{short_device_name(device_full_name)}_{short_model_name(model_arg)}"


def get_device_name(device: str) -> tuple[str, str, str]:
    """Validate/resolve device and return (resolved_device, device_full_name, cpu_full_name)."""
    import openvino as ov

    core = ov.Core()
    cpu_full_name = core.get_property("CPU", "FULL_DEVICE_NAME")
    available = core.available_devices

    if device == "GPU":
        gpu_devices = [d for d in available if d.upper().startswith("GPU")]
        intel_gpu_devices: list[tuple[str, str]] = []
        for gpu_dev in gpu_devices:
            gpu_full_name = core.get_property(gpu_dev, "FULL_DEVICE_NAME")
            if "intel" in gpu_full_name.lower():
                intel_gpu_devices.append((gpu_dev, gpu_full_name))

        if not intel_gpu_devices:
            gpu_listing: list[str] = []
            for gpu_dev in gpu_devices:
                gpu_full_name = core.get_property(gpu_dev, "FULL_DEVICE_NAME")
                gpu_listing.append(f"{gpu_dev} ({gpu_full_name})")
            listing = ", ".join(gpu_listing) if gpu_listing else "none"
            sys.exit(
                "Device GPU requested, but no Intel GPU was found. "
                f"Available GPU devices: {listing}"
            )

        resolved_device, full_name = intel_gpu_devices[0]
        if resolved_device != device:
            print(f"Resolved device alias {device} -> {resolved_device}")
    elif device != "CPU":
        if device not in available:
            sys.exit(
                f"Device {device} is not available. "
                f"Available devices: {', '.join(available)}"
            )
        resolved_device = device
        full_name = core.get_property(device, "FULL_DEVICE_NAME")
        if device.upper().startswith("GPU") and "intel" not in full_name.lower():
            sys.exit(f"Device {device} is not an Intel GPU: {full_name}")
        if not device.upper().startswith("GPU") and "intel" not in full_name.lower():
            sys.exit(f"Device {device} is not Intel: {full_name}")
    else:
        resolved_device = device
        full_name = cpu_full_name

    print(f"Device {resolved_device}: {full_name}")
    return resolved_device, full_name, cpu_full_name


def load_audio(path: str) -> tuple[np.ndarray, float]:
    """Load a WAV file as 16 kHz float32 mono and return (samples, duration_s)."""
    import librosa

    samples, _ = librosa.load(path, sr=16000, mono=True)
    duration_s = len(samples) / 16000
    return samples, duration_s


def build_pipeline(model_path: str, device: str, cache_dir: str | None,
                   static: bool = False, word_timestamps: bool = False,
                   weight_less_caching: bool = False,
                   force_plugin_npu_compiler: bool = False):
    """Construct the WhisperPipeline and return (pipe, load_ms)."""
    import openvino_genai as ov_genai

    props: dict = {}
    if cache_dir:
        if device == "CPU":
            print("WARNING: CACHE_DIR on CPU may crash — "
                  "see https://github.com/openvinotoolkit/openvino/issues/35379")
        props["CACHE_DIR"] = cache_dir
    if weight_less_caching:
        props["CACHE_MODE"] = "OPTIMIZE_SIZE"
    if force_plugin_npu_compiler and device.split(".")[0] == "NPU":
        props["NPU_COMPILER_TYPE"] = "PLUGIN"
    if static:
        props["STATIC_PIPELINE"] = True

    kwargs: dict = {}
    if word_timestamps:
        kwargs["word_timestamps"] = True

    t0 = time.perf_counter()
    try:
        pipe = ov_genai.WhisperPipeline(model_path, device, **kwargs, **props)
    except RuntimeError as exc:
        if weight_less_caching and "CACHE_MODE" in str(exc):
            sys.exit(
                f"Weightless caching (-wl) requested, but CACHE_MODE is not "
                f"supported by device/plugin '{device}' in this OpenVINO build.\n"
                f"Original error: {exc}"
            )
        raise
    load_ms = (time.perf_counter() - t0) * 1000

    return pipe, load_ms


def run_inference(pipe, samples: np.ndarray, word_timestamps: bool = False,
                  initial_prompt: str | None = None,
                  hotwords: str | None = None):
    """Run timed transcription and return (result, inference_ms)."""
    config = pipe.get_generation_config()
    config.task = "transcribe"
    config.return_timestamps = True
    if word_timestamps:
        config.word_timestamps = True
    if initial_prompt:
        config.initial_prompt = initial_prompt
    if hotwords:
        config.hotwords = hotwords
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
        help="Inference device: CPU, GPU, GPU.0, GPU.1, NPU, etc.",
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="OpenVINO compilation cache directory",
    )
    parser.add_argument(
        "--static", action="store_true",
        help="Use STATIC_PIPELINE (required for openvino-genai <= 2026.1, not required with latest nightly)",
    )
    parser.add_argument(
        "--word_timestamps", action="store_true",
        help="Enable word-level timestamps in transcription output",
    )
    parser.add_argument(
        "--weight-less-caching", "-wl", action="store_true",
        help="Enable CACHE_MODE=OPTIMIZE_SIZE (device/plugin support varies)",
    )
    parser.add_argument(
        "--force-plugin-npu-compiler", "-fpc", action="store_true",
        help="Force NPU_COMPILER_TYPE=PLUGIN on NPU (independent of -wl)",
    )
    parser.add_argument(
        "--initial_prompt", default=None,
        help="Initial prompt added to context of the first window",
    )
    parser.add_argument(
        "--hotwords", default=None,
        help="Hotwords added to context of all windows",
    )
    args = parser.parse_args()

    requested_device = args.device.upper()

    # --- Validate device --------------------------------------------------
    device, device_full_name, cpu_full_name = get_device_name(requested_device)
    if args.force_plugin_npu_compiler and device.split(".")[0] != "NPU":
        sys.exit("--force-plugin-npu-compiler/-fpc can only be used with NPU")
    if args.force_plugin_npu_compiler:
        print("Forcing NPU compiler type to PLUGIN (NPU_COMPILER_TYPE=PLUGIN).")
        print("This should always work on NPU driver >= 2565 according to "
              "https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md#ovintel_npucompiler_type")
        try:
            import openvino as ov
            npu_driver_version = ov.Core().get_property(device, "NPU_DRIVER_VERSION")
            print(f"Detected NPU driver version: {npu_driver_version}")
        except Exception as exc:
            print(f"Detected NPU driver version: <unavailable: {exc}>")

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
    if args.weight_less_caching and not args.cache_dir:
        sys.exit("--weight-less-caching/-wl requires --cache_dir")

    if args.weight_less_caching:
        print(f"Weightless caching enabled on {device} (CACHE_MODE=OPTIMIZE_SIZE)")

    print(f"Loading model on {device} …")
    pipe, load_ms = build_pipeline(model_path, device, args.cache_dir, args.static,
                                    args.word_timestamps, args.weight_less_caching,
                                    args.force_plugin_npu_compiler)
    print(f"Model loaded in {load_ms:.0f} ms")

    # --- Timed inference ---------------------------------------------------
    print("Running timed inference …")
    result, inference_ms = run_inference(pipe, samples, args.word_timestamps,
                                         args.initial_prompt, args.hotwords)

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

    words = getattr(result, "words", None) or []

    # Console-truncated word timestamps
    CONSOLE_MAX_WORDS = 30
    console_word_lines: list[str] = []
    if words:
        for w in words[:CONSOLE_MAX_WORDS]:
            console_word_lines.append(
                f"  [{w.start_ts:6.2f}s -> {w.end_ts:6.2f}s]  {w.word}")
        if len(words) > CONSOLE_MAX_WORDS:
            console_word_lines.append(
                f"  ... ({len(words) - CONSOLE_MAX_WORDS} more words)")

    header_lines = [
        "=" * 60,
        f"{'Command:':<18}{cmdline}",
        f"{'Platform:':<18}{platform.platform()}",
        f"{'Device:':<18}{requested_device if requested_device == device else f'{requested_device} -> {device}'} ({device_full_name})",
        f"{'Model:':<18}{model_label}",
        f"{'Audio:':<18}{os.path.basename(args.audio)} ({duration_s:.1f}s)",
    ]
    if args.initial_prompt:
        header_lines.append(f"{'Initial prompt:':<18}{args.initial_prompt}")
    if args.hotwords:
        header_lines.append(f"{'Hotwords:':<18}{args.hotwords}")
    header_lines += [
        "",
        f"{'Load+compile:':<18}{load_ms:>8.0f} ms",
        f"{'Inference:':<18}{inference_ms:>8.0f} ms  ({realtime_factor:.1f}x realtime)",
        f"{'Total:':<18}{total_ms:>8.0f} ms",
    ]

    # --- Console report (truncated) ----------------------------------------
    console_lines = header_lines + [
        "",
        "Transcript (first 200 chars):",
        f'  "{preview}"',
    ]
    if console_word_lines:
        console_lines.append("")
        console_lines.append(
            f"Word timestamps (first {min(len(words), CONSOLE_MAX_WORDS)} words):")
        console_lines.extend(console_word_lines)
    console_lines.append("=" * 60)

    print()
    print("\n".join(console_lines))

    # --- File report (full transcript + all word timestamps) ---------------
    file_lines = header_lines + [
        "",
        "Full transcript:",
        f'  "{text}"',
    ]
    if words:
        file_lines.append("")
        file_lines.append(f"Word timestamps ({len(words)} words):")
        for w in words:
            file_lines.append(
                f"  [{w.start_ts:6.2f}s -> {w.end_ts:6.2f}s]  {w.word}")
    file_lines.append("=" * 60)
    file_report = "\n".join(file_lines)

    # --- Save to file ------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    stem = result_filename_stem(device, cpu_full_name, model_label)
    result_path = os.path.join(results_dir, f"{stem}.txt")
    counter = 2
    while os.path.exists(result_path):
        result_path = os.path.join(results_dir, f"{stem}_{counter}.txt")
        counter += 1

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(file_report + "\n")

    print(f"\nResult saved to {result_path}")


if __name__ == "__main__":
    main()
