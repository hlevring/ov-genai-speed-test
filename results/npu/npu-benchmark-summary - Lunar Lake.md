# NPU Whisper Benchmark Summary (Lunar Lake)

## Test Environment


|                     |                                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Models**          | https://huggingface.co/hlevring/ov-whisper_small-int8-2026.0.0, https://huggingface.co/hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 |
| **Test script**     | https://github.com/hlevring/ov-genai-speed-test                                                                                         |
| **Platform**        | Windows 11 (10.0.26200)                                                                                     |
| **Processor**       | Intel Core Ultra 9 288V (from benchmark result filename stem / CPU device name)                             |
| **NPU**             | Intel(R) AI Boost                                                                                           |
| **OpenVINO**        | `2026.2.0.dev20260511`                                                                                      |
| **OpenVINO GenAI**  | `2026.2.0.0.dev20260511` (nightly wheels)                                                                    |
| **Audio**           | `demoB108s.wav` (108.6s, mono, 16 kHz)                                                                      |
| **Word timestamps** | Enabled                                                                                                     |
| **Cache directory** | `./cache` (shared across all runs; within each model/pipeline pair, run 1 = cold cache, run 2 = warm cache) |


---

## Whisper Small Int8 — Stateful Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total     |
| ---------- | ------------ | --------- | --------------- | --------- |
| Cold cache | 70,500 ms    | 11,660 ms | 9.3x            | 82,160 ms |
| Warm cache | 2,644 ms     | 11,188 ms | 9.7x            | 13,832 ms |


---

## Whisper Small Int8 — Static Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --static --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total     |
| ---------- | ------------ | --------- | --------------- | --------- |
| Cold cache | 70,223 ms    | 11,511 ms | 9.4x            | 81,734 ms |
| Warm cache | 2,303 ms     | 11,215 ms | 9.7x            | 13,519 ms |


---

## Whisper Large V3 Turbo Int8 — Stateful Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total      |
| ---------- | ------------ | --------- | --------------- | ---------- |
| Cold cache | 158,423 ms   | 9,475 ms  | 11.5x           | 167,898 ms |
| Warm cache | 2,661 ms     | 9,323 ms  | 11.6x           | 11,984 ms  |


---

## Whisper Large V3 Turbo Int8 — Static Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --static --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total      |
| ---------- | ------------ | --------- | --------------- | ---------- |
| Cold cache | 167,769 ms   | 9,802 ms  | 11.1x           | 177,571 ms |
| Warm cache | 2,024 ms     | 8,965 ms  | 12.1x           | 10,989 ms  |


---

## Comparison (Warm Cache)


| Model               | Pipeline | Load+Compile | Inference | Realtime Factor | Total     |
| ------------------- | -------- | ------------ | --------- | --------------- | --------- |
| Small Int8          | Stateful | 2,644 ms     | 11,188 ms | 9.7x            | 13,832 ms |
| Small Int8          | Static   | 2,303 ms     | 11,215 ms | 9.7x            | 13,519 ms |
| Large V3 Turbo Int8 | Stateful | 2,661 ms     | 9,323 ms  | 11.6x           | 11,984 ms |
| Large V3 Turbo Int8 | Static   | 2,024 ms     | 8,965 ms  | 12.1x           | 10,989 ms |


## Comparison (Cold Cache)


| Model               | Pipeline | Load+Compile | Inference | Realtime Factor | Total      |
| ------------------- | -------- | ------------ | --------- | --------------- | ---------- |
| Small Int8          | Stateful | 70,500 ms    | 11,660 ms | 9.3x            | 82,160 ms  |
| Small Int8          | Static   | 70,223 ms    | 11,511 ms | 9.4x            | 81,734 ms  |
| Large V3 Turbo Int8 | Stateful | 158,423 ms   | 9,475 ms  | 11.5x           | 167,898 ms |
| Large V3 Turbo Int8 | Static   | 167,769 ms   | 9,802 ms  | 11.1x           | 177,571 ms |

