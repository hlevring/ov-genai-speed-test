# GPU Weightless Cache Benchmark Summary (Panther Lake, _2)

## Test Environment

|                     |                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------- |
| **Platform**        | Windows 10 (10.0.26200)                                                                 |
| **Processor**       | Intel Core Ultra X7 358H                                                                |
| **GPU**             | Intel(R) Arc(TM) B390 GPU (iGPU) (`GPU` resolved to `GPU.0`)                           |
| **OpenVINO**        | `2026.2.0-21894-5ec01181735-releases/2026/2`                                            |
| **OpenVINO GenAI**  | `2026.2.0.0-3119-1550089e312`                                                           |
| **Audio**           | `demoB108s.wav` (108.6s, mono, 16 kHz)                                                  |
| **Word timestamps** | Enabled                                                                                 |
| **Cache directory** | `./cache` (cleared before every cold run scenario)                                      |
| **Models**          | `hlevring/ov-whisper_small-int8-2026.0.0`, `hlevring/ov-whisper_large_v3_turbo-int8-2026.0.0` |

---

## Whisper Small Int8

### With `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total    |
| ---------- | ------------ | --------- | --------------- | -------- |
| Cold cache | 2,057 ms     | 4,315 ms  | 25.2x           | 6,372 ms |
| Warm cache | 424 ms       | 3,740 ms  | 29.0x           | 4,165 ms |

Cache artifacts after cold run:
- File count: `69`
- Total size: `64,177,411` bytes (`61.204 MB`)

### Without `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total    |
| ---------- | ------------ | --------- | --------------- | -------- |
| Cold cache | 1,988 ms     | 4,094 ms  | 26.5x           | 6,083 ms |
| Warm cache | 436 ms       | 3,788 ms  | 28.7x           | 4,224 ms |

Cache artifacts after cold run:
- File count: `69`
- Total size: `263,597,672` bytes (`251.386 MB`)

### Small Comparison (`-wl` vs no `-wl`)

- Cache size delta: `-199,420,261` bytes (`-190.182 MB`, ~75.7% smaller with `-wl`)
- Cold total delta: `6,372 ms` vs `6,083 ms` (`+289 ms`, `-wl` slower)
- Warm total delta: `4,165 ms` vs `4,224 ms` (`-59 ms`, `-wl` faster)

---

## Whisper Large V3 Turbo Int8

### With `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total    |
| ---------- | ------------ | --------- | --------------- | -------- |
| Cold cache | 2,542 ms     | 3,145 ms  | 34.5x           | 5,687 ms |
| Warm cache | 681 ms       | 2,683 ms  | 40.5x           | 3,364 ms |

Cache artifacts after cold run:
- File count: `77`
- Total size: `207,040,823` bytes (`197.450 MB`)

### Without `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total    |
| ---------- | ------------ | --------- | --------------- | -------- |
| Cold cache | 2,866 ms     | 3,129 ms  | 34.7x           | 5,995 ms |
| Warm cache | 562 ms       | 2,669 ms  | 40.7x           | 3,231 ms |

Cache artifacts after cold run:
- File count: `77`
- Total size: `837,198,255` bytes (`798.414 MB`)

### V3 Comparison (`-wl` vs no `-wl`)

- Cache size delta: `-630,157,432` bytes (`-600.964 MB`, ~75.3% smaller with `-wl`)
- Cold total delta: `5,687 ms` vs `5,995 ms` (`-308 ms`, `-wl` faster)
- Warm total delta: `3,364 ms` vs `3,231 ms` (`+133 ms`, `-wl` slower)

---

## Overall Findings

- `-wl` still significantly reduces GPU cache footprint for both models (~75%).
- Runtime impact remains small and mixed (better in some runs, slightly worse in others).
