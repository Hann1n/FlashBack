<div align="center">

<br>

# FlashBack

**Rewinding Fire to Its Origin**

[![NVIDIA Cosmos](https://img.shields.io/badge/NVIDIA-Cosmos--Reason2-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://build.nvidia.com/nvidia/cosmos-reason2)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Cookoff](https://img.shields.io/badge/Cosmos_Cookoff-2026-FF6F00?style=for-the-badge)](https://www.nvidia.com/en-us/ai/cosmos-cookoff/)

<br>

감시 영상에서 **화재 발원지를 물리적으로 역추적**하는 AI 시스템
<br>
기존 화재 감지가 *"불이 있는가?"* 를 묻는다면,
<br>
FlashBack은 ***"불이 어디서 시작되어, 어떻게 퍼졌는가?"*** 를 답합니다.

<br>

</div>

## Demo

https://github.com/Hann1n/flashback/raw/master/demo/demo.mp4

<br>

## How It Works

<br>

<div align="center">

```
  📹 감시 영상                  🧠 Cosmos-Reason2                📍 발원지 추론
 ┌───────────┐              ┌──────────────────┐             ┌────────────────┐
 │           │    fps=1     │                  │   (x, y)    │  Origin Point  │
 │  MP4/JPG  │ ──────────▶ │  Physics-Aware   │ ──────────▶ │  + Spread Dir  │
 │  Frames   │   sampling  │  Video Reasoning │   coords    │  + Arrows      │
 └───────────┘              └──────────────────┘             └───────┬────────┘
                                    │                                │
                             ┌──────▼──────┐                ┌───────▼────────┐
                             │   Chain of  │                │  Lucas-Kanade  │
                             │   Thought   │                │  Optical Flow  │
                             │             │                │                │
                             │  연소 물리학  │                │  프레임 간 추적  │
                             │  대류 패턴   │                │  카메라 보정    │
                             │  연기 역학   │                │  안정적 마커    │
                             └─────────────┘                └───────┬────────┘
                                                                    │
                                                            ┌───────▼────────┐
                                                            │   📊 Output    │
                                                            │  Video + Image │
                                                            │  Dashboard     │
                                                            └────────────────┘
```

</div>

<br>

FlashBack은 **세 단계**로 작동합니다:

### 1. Physics-Aware Prompting

Cosmos-Reason2를 **화재 물리학 전문가**로 프롬프팅합니다.
모델은 영상을 프레임 단위로 분석하며, 연소 물리학에 기반한 Chain-of-Thought 추론을 수행합니다.

> *"화염이 외부로 확산되고 연기가 상부에 축적되는 패턴은 전형적인 성장기 화재를 나타냅니다.*
> *발원지는 연료원이 집중된 하단부이며, 열대류에 의해 상향 확산되고 있습니다."*
> — Cosmos-Reason2 Chain-of-Thought 추론 예시

### 2. Coordinate-Based Origin Tracing

모델이 **정규화 좌표 (x, y)** 로 발원지를 출력합니다.
텍스트 설명("온실 좌측 하단")과 함께 정밀 좌표(0.25, 0.75)를 동시에 추론하여,
원본 프레임 위에 **발원지 마커 + 확산 방향 화살표**를 시각적으로 오버레이합니다.

### 3. Optical Flow Tracking

**Lucas-Kanade 광학 흐름**으로 발원지 좌표를 전체 프레임에 걸쳐 추적합니다.
카메라가 움직여도 발원지 마커가 정확한 위치에 고정되어,
데모 영상에서 화재 확산 과정을 실시간으로 시각화합니다.

<br>

### Physics Reasoning

| Principle | What the model analyzes | How it traces the origin |
|:----------|:------------------------|:-------------------------|
| **Convection** | 열기류 상승, 연기 운반 경로 | 연기 축적점 아래 = 발원지 |
| **Combustion** | 연료 밀도 ↔ 화염 강도 | 최초 연료 집중 지점 특정 |
| **Propagation** | 열전달에 의한 외부 확산 | 확산 중심점 역추적 |
| **Smoke Color** | 연기 색상 → 연소 물질 | 발원 위치 + 연료 추정 |
| **Temporal** | 발화 → 성장 → 최성기 → 쇠퇴 | 프레임 변화로 방향 역산 |

<br>

## Results

> **Cosmos-Reason2-2B** &nbsp;|&nbsp; 11 scenes &nbsp;|&nbsp; FLAME 5 / SMOKE 2 / NORMAL 4

| Metric | Score |
|:-------|------:|
| Fire Origin Tracing | **100%** |
| Temporal Reasoning | **100%** |
| Spread Direction | **85.7%** |
| Hazard Detection | **71.4%** |

<br>

## Origin Visualization

<table>
<tr>
<td width="50%"><img src="assets/origin_flame.jpg" width="100%"><br><sub><b>Scene 0087 — FLAME</b> · 온실 화재 발원지 + 확산 방향</sub></td>
<td width="50%"><img src="assets/origin_smoke.jpg" width="100%"><br><sub><b>Scene 0096 — SMOKE</b> · 연기 확산에서 발원지 역추적</sub></td>
</tr>
</table>

<img src="assets/temporal_strip.jpg" width="100%">
<sub>Temporal progression — 시간 순서 프레임에서 발원지(빨간 원)의 위치 변화</sub>

<br>

## Quick Start

```bash
git clone https://github.com/Hann1n/flashback.git
cd flashback

python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

```bash
# Full pipeline
python run.py

# Or step by step
python src/detection.py          # Cosmos-Reason2 inference (GPU)
python src/visualize.py          # Origin overlay images
python src/dashboard.py          # HTML dashboard
python src/video.py              # Demo video with optical flow
streamlit run src/app.py         # Interactive dashboard
```

<br>

## Project Structure

```
flashback/
├── run.py                     # Pipeline entry point
├── src/
│   ├── inference.py           # Cosmos-Reason2 model wrapper
│   ├── detection.py           # Fire detection + origin inference
│   ├── new_data.py            # New dataset inference pipeline
│   ├── visualize.py           # Origin marker visualization
│   ├── dashboard.py           # HTML dashboard generator
│   ├── video.py               # Demo video (optical flow tracking)
│   ├── fiftyone_builder.py    # FiftyOne dataset builder
│   └── app.py                 # Streamlit interactive dashboard
├── reports/                   # Inference results (JSON)
├── demo/                      # Demo video
├── assets/                    # README images
├── requirements.txt
└── README.md
```

<br>

## Technical Stack

| | |
|:--|:--|
| **Model** | Cosmos-Reason2-2B (Qwen3VL) |
| **Tracking** | Lucas-Kanade Optical Flow |
| **Video** | PyAV backend (Windows FFmpeg workaround) |
| **Inference** | fps=1, temp=0.6, CoT reasoning enabled |
| **Visualization** | OpenCV, Plotly, Chart.js, Streamlit |

<br>

<div align="center">

Built for [NVIDIA Cosmos Cookoff 2026](https://www.nvidia.com/en-us/ai/cosmos-cookoff/)
<br>
Uses [Cosmos-Reason2](https://build.nvidia.com/nvidia/cosmos-reason2) under NVIDIA Open Model License

</div>
