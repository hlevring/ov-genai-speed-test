# CPU Weightless Cache Benchmark Summary (Panther Lake, _2)

## Test Environment

|                     |                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------- |
| **Platform**        | Windows 10 (10.0.26200)                                                                 |
| **Processor**       | Intel Core Ultra X7 358H                                                                |
| **Device**          | CPU                                                                                     |
| **OpenVINO**        | `2026.2.0-21894-5ec01181735-releases/2026/2`                                            |
| **OpenVINO GenAI**  | `2026.2.0.0-3119-1550089e312`                                                           |
| **Audio**           | `demoB108s.wav` (108.6s, mono, 16 kHz)                                                  |
| **Word timestamps** | Enabled                                                                                 |
| **Cache directory** | `./cache` (cleared before every cold run scenario)                                      |

---

## Whisper Small Int8

### With `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total    |
| ---------- | ------------ | --------- | --------------- | -------- |
| Cold cache | 1,106 ms     | 5,882 ms  | 18.5x           | 6,988 ms |
| Warm cache | 869 ms       | 5,800 ms  | 18.7x           | 6,669 ms |

Cache artifacts after cold run:
- File count: `2`
- Total size: `10,313,692` bytes (`9.836 MB`)

### Without `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total    |
| ---------- | ------------ | --------- | --------------- | -------- |
| Cold cache | 1,200 ms     | 5,746 ms  | 18.9x           | 6,946 ms |
| Warm cache | 924 ms       | 6,073 ms  | 17.9x           | 6,997 ms |

Cache artifacts after cold run:
- File count: `2`
- Total size: `255,276,554` bytes (`243.451 MB`)

### Small Comparison (`-wl` vs no `-wl`)

- Cache size delta: `-244,962,862` bytes (`-233.615 MB`, ~96.0% smaller with `-wl`)
- Cold total delta: `6,988 ms` vs `6,946 ms` (`+42 ms`, `-wl` slower)
- Warm total delta: `6,669 ms` vs `6,997 ms` (`-328 ms`, `-wl` faster)

---

## Whisper Large V3 Turbo Int8

### With `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total     |
| ---------- | ------------ | --------- | --------------- | --------- |
| Cold cache | 1,259 ms     | 15,024 ms | 7.2x            | 16,282 ms |
| Warm cache | 618 ms       | 14,924 ms | 7.3x            | 15,542 ms |

Cache artifacts after cold run:
- File count: `2`
- Total size: `25,133,685` bytes (`23.969 MB`)

### Without `-wl`

| Run        | Load+Compile | Inference | Realtime Factor | Total     |
| ---------- | ------------ | --------- | --------------- | --------- |
| Cold cache | 1,753 ms     | 15,011 ms | 7.2x            | 16,763 ms |
| Warm cache | 708 ms       | 14,985 ms | 7.2x            | 15,693 ms |

Cache artifacts after cold run:
- File count: `2`
- Total size: `837,457,944` bytes (`798.662 MB`)

### V3 Comparison (`-wl` vs no `-wl`)

- Cache size delta: `-812,324,259` bytes (`-774.693 MB`, ~97.0% smaller with `-wl`)
- Cold total delta: `16,282 ms` vs `16,763 ms` (`-481 ms`, `-wl` faster)
- Warm total delta: `15,542 ms` vs `15,693 ms` (`-151 ms`, `-wl` faster)

---

## Overall Findings

- CPU cache footprint reduction with `-wl` is very large for both models (96%+).
- Runtime is similar or slightly improved in most warm/cold comparisons in this rerun.
