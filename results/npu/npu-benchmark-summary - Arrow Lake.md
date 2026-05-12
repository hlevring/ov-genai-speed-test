# NPU Whisper Benchmark Summary (Arrow Lake)

## Test Environment


|                     |                                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Models**          | https://huggingface.co/hlevring/ov-whisper_small-int8-2026.0.0, https://huggingface.co/hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 |
| **Test script**     | https://github.com/hlevring/ov-genai-speed-test                                                                                         |
| **Platform**        | Windows 10 (10.0.26200)                                                                                     |
| **Processor**       | Intel Core Ultra 7 255H (Arrow Lake Laptop)                                                                 |
| **NPU**             | Intel(R) AI Boost                                                                                           |
| **OpenVINO**        | `2026.2.0.dev20260509`                                                                                      |
| **OpenVINO GenAI**  | `2026.2.0.0.dev20260509` (nightly wheels)                                                                   |
| **Audio**           | `demoB108s.wav` (108.6s, mono, 16 kHz)                                                                      |
| **Word timestamps** | Enabled                                                                                                     |
| **Cache directory** | `./cache` (cleared before starting; within each model/pipeline pair, run 1 = cold cache, run 2 = warm cache) |


---

## Whisper Small Int8 — Stateful Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --word_timestamps
```


| Run        | Load+Compile | Inference  | Realtime Factor | Total      |
| ---------- | ------------ | ---------- | --------------- | ---------- |
| Cold cache | 75,179 ms    | 15,922 ms  | 6.8x            | 91,101 ms  |
| Warm cache | 2,351 ms     | 15,433 ms  | 7.0x            | 17,784 ms  |


---

## Whisper Small Int8 — Static Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --static --word_timestamps
```


| Run        | Load+Compile | Inference  | Realtime Factor | Total      |
| ---------- | ------------ | ---------- | --------------- | ---------- |
| Cold cache | 79,234 ms    | 15,459 ms  | 7.0x            | 94,694 ms  |
| Warm cache | 1,352 ms     | 14,947 ms  | 7.3x            | 16,299 ms  |


---

## Whisper Large V3 Turbo Int8 — Stateful Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --word_timestamps
```


| Run        | Load+Compile | Inference  | Realtime Factor | Total      |
| ---------- | ------------ | ---------- | --------------- | ---------- |
| Cold cache | 237,717 ms   | 17,506 ms  | 6.2x            | 255,223 ms |
| Warm cache | 6,129 ms     | 16,575 ms  | 6.6x            | 22,704 ms  |


---

## Whisper Large V3 Turbo Int8 — Static Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --static --word_timestamps
```


| Run        | Load+Compile | Inference  | Realtime Factor | Total      |
| ---------- | ------------ | ---------- | --------------- | ---------- |
| Cold cache | 279,050 ms   | 19,464 ms  | 5.6x            | 298,514 ms |
| Warm cache | 4,407 ms     | 19,243 ms  | 5.6x            | 23,649 ms  |


---

## Comparison (Warm Cache)


| Model               | Pipeline | Load+Compile | Inference  | Realtime Factor | Total     |
| ------------------- | -------- | ------------ | ---------- | --------------- | --------- |
| Small Int8          | Stateful | 2,351 ms     | 15,433 ms  | 7.0x            | 17,784 ms |
| Small Int8          | Static   | 1,352 ms     | 14,947 ms  | 7.3x            | 16,299 ms |
| Large V3 Turbo Int8 | Stateful | 6,129 ms     | 16,575 ms  | 6.6x            | 22,704 ms |
| Large V3 Turbo Int8 | Static   | 4,407 ms     | 19,243 ms  | 5.6x            | 23,649 ms |


## Comparison (Cold Cache)


| Model               | Pipeline | Load+Compile | Inference  | Realtime Factor | Total      |
| ------------------- | -------- | ------------ | ---------- | --------------- | ---------- |
| Small Int8          | Stateful | 75,179 ms    | 15,922 ms  | 6.8x            | 91,101 ms  |
| Small Int8          | Static   | 79,234 ms    | 15,459 ms  | 7.0x            | 94,694 ms  |
| Large V3 Turbo Int8 | Stateful | 237,717 ms   | 17,506 ms  | 6.2x            | 255,223 ms |
| Large V3 Turbo Int8 | Static   | 279,050 ms   | 19,464 ms  | 5.6x            | 298,514 ms |

