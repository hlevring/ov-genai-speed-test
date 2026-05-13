# NPU Weightless Cache Benchmark Summary (Panther Lake, _2)

## Test Environment

|                     |                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------- |
| **Platform**        | Windows 10 (10.0.26200)                                                                 |
| **Processor**       | Intel Core Ultra X7 358H                                                                |
| **NPU**             | Intel(R) AI Boost                                                                       |
| **OpenVINO**        | `2026.2.0-21894-5ec01181735-releases/2026/2`                                            |
| **OpenVINO GenAI**  | `2026.2.0.0-3119-1550089e312`                                                           |
| **Audio**           | `demoB108s.wav` (108.6s, mono, 16 kHz)                                                  |
| **Word timestamps** | Enabled                                                                                 |
| **Pipeline mode**   | Stateful only (no `--static`)                                                           |
| **Cache directory** | `./cache` (cleared before every cold run scenario)                                      |

---

## Whisper Small Int8

### With `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total      |
| ---------- | ------------ | --------- | --------------- | ---------- |
| Cold cache | 78,072 ms    | 10,214 ms | 10.6x           | 88,286 ms  |
| Warm cache | 1,470 ms     | 9,463 ms  | 11.5x           | 10,933 ms  |

Cache artifacts after cold run:
- File count: `2`
- Total size: `770,794,650` bytes (`735.087 MB`)

### Without `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total      |
| ---------- | ------------ | --------- | --------------- | ---------- |
| Cold cache | 74,803 ms    | 9,611 ms  | 11.3x           | 84,413 ms  |
| Warm cache | 1,488 ms     | 9,508 ms  | 11.4x           | 10,996 ms  |

Cache artifacts after cold run:
- File count: `2`
- Total size: `910,013,531` bytes (`867.857 MB`)

### Small Comparison (`-wl` vs no `-wl`)

- Cache size delta: `-139,218,881` bytes (`-132.770 MB`, ~15.3% smaller with `-wl`)
- Cold total delta: `88,286 ms` vs `84,413 ms` (`+3,873 ms`, `-wl` slower)
- Warm total delta: `10,933 ms` vs `10,996 ms` (`-63 ms`, `-wl` faster)

---

## Whisper Large V3 Turbo Int8

### With `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total       |
| ---------- | ------------ | --------- | --------------- | ----------- |
| Cold cache | 656,803 ms   | 7,419 ms  | 14.6x           | 664,222 ms  |
| Warm cache | 2,227 ms     | 7,604 ms  | 14.3x           | 9,830 ms    |

Cache artifacts after cold run:
- File count: `2`
- Total size: `944,140,213` bytes (`900.402 MB`)

### Without `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total       |
| ---------- | ------------ | --------- | --------------- | ----------- |
| Cold cache | 627,679 ms   | 8,106 ms  | 13.4x           | 635,784 ms  |
| Warm cache | 1,742 ms     | 7,332 ms  | 14.8x           | 9,075 ms    |

Cache artifacts after cold run:
- File count: `2`
- Total size: `2,148,843,366` bytes (`2,049.297 MB`)

### V3 Comparison (`-wl` vs no `-wl`)

- Cache size delta: `-1,204,703,153` bytes (`-1,148.895 MB`, ~56.1% smaller with `-wl`)
- Cold total delta: `664,222 ms` vs `635,784 ms` (`+28,438 ms`, `-wl` slower)
- Warm total delta: `9,830 ms` vs `9,075 ms` (`+755 ms`, `-wl` slower)

---

## Overall Findings

- On NPU, `-wl` reduces cache size clearly, especially on V3.
- In this rerun, `-wl` tends to add overhead on NPU runtime for V3, while Small warm run remains similar.
