# NPU Whisper Benchmark Summary (Panther Lake)

## Test Environment


|                     |                                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Models**          | https://huggingface.co/hlevring/ov-whisper_small-int8-2026.0.0, https://huggingface.co/hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 |
| **Test script**     | https://github.com/hlevring/ov-genai-speed-test                                                                                         |
| **Platform**        | Windows 10 (10.0.26200)                                                                                     |
| **Processor**       | Intel Core Ultra X7 358H (from benchmark result stem / CPU device name)                                     |
| **NPU**             | Intel(R) AI Boost                                                                                           |
| **OpenVINO**        | `2026.2.0.dev20260509`                                                                                      |
| **OpenVINO GenAI**  | `2026.2.0.0.dev20260509` (nightly wheels)                                                                   |
| **Audio**           | `demoB108s.wav` (108.6s, mono, 16 kHz)                                                                      |
| **Word timestamps** | Enabled                                                                                                     |
| **Cache directory** | `./cache` (shared across all runs; within each model/pipeline pair, run 1 = cold cache, run 2 = warm cache) |


---

## Whisper Small Int8 — Stateful Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total      |
| ---------- | ------------ | --------- | --------------- | ---------- |
| Cold cache | 103,617 ms   | 9,788 ms  | 11.1x           | 113,405 ms |
| Warm cache | 1,806 ms     | 9,466 ms  | 11.5x           | 11,272 ms  |


---

## Whisper Small Int8 — Static Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_small-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --static --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total     |
| ---------- | ------------ | --------- | --------------- | --------- |
| Cold cache | 71,063 ms    | 9,763 ms  | 11.1x           | 80,825 ms |
| Warm cache | 884 ms       | 9,780 ms  | 11.1x           | 10,664 ms |


---

## Whisper Large V3 Turbo Int8 — Stateful Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total      |
| ---------- | ------------ | --------- | --------------- | ---------- |
| Cold cache | 659,428 ms   | 7,566 ms  | 14.4x           | 666,994 ms |
| Warm cache | 1,861 ms     | 7,710 ms  | 14.1x           | 9,571 ms   |


---

## Whisper Large V3 Turbo Int8 — Static Pipeline

```
python -m ov_asr_speed --model hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0 --audio demoB108s.wav -d npu --cache_dir cache --static --word_timestamps
```


| Run        | Load+Compile | Inference | Realtime Factor | Total      |
| ---------- | ------------ | --------- | --------------- | ---------- |
| Cold cache | 674,299 ms   | 7,989 ms  | 13.6x           | 682,288 ms |
| Warm cache | 1,275 ms     | 7,660 ms  | 14.2x           | 8,934 ms   |


---

## Comparison (Warm Cache)


| Model               | Pipeline | Load+Compile | Inference | Realtime Factor | Total     |
| ------------------- | -------- | ------------ | --------- | --------------- | --------- |
| Small Int8          | Stateful | 1,806 ms     | 9,466 ms  | 11.5x           | 11,272 ms |
| Small Int8          | Static   | 884 ms       | 9,780 ms  | 11.1x           | 10,664 ms |
| Large V3 Turbo Int8 | Stateful | 1,861 ms     | 7,710 ms  | 14.1x           | 9,571 ms  |
| Large V3 Turbo Int8 | Static   | 1,275 ms     | 7,660 ms  | 14.2x           | 8,934 ms  |


## Comparison (Cold Cache)


| Model               | Pipeline | Load+Compile | Inference | Realtime Factor | Total      |
| ------------------- | -------- | ------------ | --------- | --------------- | ---------- |
| Small Int8          | Stateful | 103,617 ms   | 9,788 ms  | 11.1x           | 113,405 ms |
| Small Int8          | Static   | 71,063 ms    | 9,763 ms  | 11.1x           | 80,825 ms  |
| Large V3 Turbo Int8 | Stateful | 659,428 ms   | 7,566 ms  | 14.4x           | 666,994 ms |
| Large V3 Turbo Int8 | Static   | 674,299 ms   | 7,989 ms  | 13.6x           | 682,288 ms |


