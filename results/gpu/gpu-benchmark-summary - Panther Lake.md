# GPU Whisper Benchmark Summary (Panther Lake)

## Test Environment


|                     |                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Platform**        | Windows 10 (10.0.26200)                                                                                      |
| **Processor**       | Intel Core Ultra X7 358H (from benchmark result stem / CPU device name)                                      |
| **GPU**             | Intel(R) Arc(TM) B390 GPU (iGPU)                                                                             |
| **OpenVINO**        | `2026.2.0-21894-5ec01181735-releases/2026/2`                                                                  |
| **OpenVINO GenAI**  | `2026.2.0.0-3119-1550089e312` (nightly wheels)                                                                |
| **Audio**           | `demoB108s.wav` (108.6s, mono, 16 kHz)                                                                       |
| **Word timestamps** | Enabled                                                                                                      |
| **Cache directory** | `./cache` (reset before each model; within each model, run 1 = cold cache, run 2 = warm cache)              |
| **Pipeline scope**  | Stateful only (`STATIC_PIPELINE` intentionally skipped for GPU)                                               |


---

## Whisper Small Int8 — Stateful Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d GPU --cache_dir cache --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total    |
| ---------- | ------------ | --------- | --------------- | -------- |
| Cold cache | 2,037 ms     | 4,071 ms  | 26.7x           | 6,108 ms |
| Warm cache | 448 ms       | 3,682 ms  | 29.5x           | 4,129 ms |


---

## Whisper Large V3 Turbo Int8 — Stateful Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 --audio demoB108s.wav -d GPU --cache_dir cache --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total     |
| ---------- | ------------ | --------- | --------------- | --------- |
| Cold cache | 13,743 ms    | 2,986 ms  | 36.4x           | 16,729 ms |
| Warm cache | 587 ms       | 2,703 ms  | 40.2x           | 3,291 ms  |


---

## Comparison (Warm Cache)


| Model               | Pipeline | Load+Compile | Inference | Realtime Factor | Total    |
| ------------------- | -------- | ------------ | --------- | --------------- | -------- |
| Small Int8          | Stateful | 448 ms       | 3,682 ms  | 29.5x           | 4,129 ms |
| Large V3 Turbo Int8 | Stateful | 587 ms       | 2,703 ms  | 40.2x           | 3,291 ms |


## Comparison (Cold Cache)


| Model               | Pipeline | Load+Compile | Inference | Realtime Factor | Total     |
| ------------------- | -------- | ------------ | --------- | --------------- | --------- |
| Small Int8          | Stateful | 2,037 ms     | 4,071 ms  | 26.7x           | 6,108 ms  |
| Large V3 Turbo Int8 | Stateful | 13,743 ms    | 2,986 ms  | 36.4x           | 16,729 ms |
