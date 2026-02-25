# FlashBack — Rewinding Fire to Its Origin

**NVIDIA Cosmos Cookoff 2026 제출작**

FlashBack은 NVIDIA Cosmos-Reason2의 물리 추론 능력을 활용하여 감시 영상에서 **화재 발원지를 역추적**하는 시스템입니다. 단순 화재 감지가 아닌, 시간에 따른 화염 확산 패턴과 연기 분산 역학을 물리적으로 분석하여 **불이 어디서 시작되었는지**를 이미지 좌표로 추론하고 시각화합니다.

## 핵심 차별점

기존 화재 감지: *"불이 있는가?"*
**FlashBack**: *"불이 어디서 시작되어, 어떻게 퍼졌는가?"*

Cosmos-Reason2가 이해하는 연소 물리학 — 대류에 의한 상승 확산, 열전달 기반 화염 전파, 연기 축적 역학 — 을 활용하여 화재를 시간 역순으로 추적, 발원지를 정확히 지목합니다.

## 물리적 추론 근거

FlashBack은 다음과 같은 물리적 원리를 기반으로 발원지를 추론합니다:

| 물리적 원리 | 설명 | 역추적 활용 |
|------------|------|------------|
| **대류 패턴** | 열기류가 상승하며 연기를 위로 운반 | 연기 축적 지점 아래가 발원지 |
| **연소 물리학** | 연료 밀도가 높은 곳에서 화염 강도가 최대 | 최초 연료 집중 지점 = 발원지 |
| **화염 전파** | 열에 의해 화염이 외부로 확산 | 확산의 중심점 역추적 |
| **연기 색상 분석** | 연료 종류에 따라 연기 색상이 다름 | 연소 물질 특성으로 발원 위치 추정 |
| **시간적 진행** | 화재 단계 (발화→성장→최성기→쇠퇴) | 프레임 간 변화로 확산 방향 역산 |
| **광학 흐름 추적** | Lucas-Kanade 알고리즘으로 프레임 간 이동 추적 | 발원지 좌표를 카메라 움직임에 맞춰 보정 |

## 결과 (Cosmos-Reason2-2B)

11개 장면 (FLAME 5 / SMOKE 2 / NORMAL 4) 에 대한 평가 결과:

| 지표 | 점수 |
|------|------|
| 위험 감지 정확도 | 71.4% |
| 발원지 추론율 | **100%** |
| 확산 방향 추론율 | 85.7% |
| 시간적 추론율 | **100%** |

- **발원지 추론율 100%**: 화재가 감지된 모든 장면에서 발원지 좌표를 성공적으로 추론
- **시간적 추론율 100%**: 모든 장면에서 시간에 따른 화재 진행 상황을 물리적으로 분석

데이터: AIHub 화재 감지 데이터셋 + 자체 수집 영상 (실내 화재, 산업 시설, 야외 화재)

## 프로젝트 구조

```
firetrace/
├── inference.py               # Cosmos-Reason2 모델 래퍼 (PyAV 백엔드)
├── fire_detection.py          # 추론 + 평가 (좌표 출력 포함)
├── firetrace_new_data.py      # 신규 데이터셋 추론
├── visualize_origin.py        # 발원지 마커 + 확산 화살표 시각화
├── firetrace_dashboard.py     # HTML 대시보드 생성기
├── firetrace_app.py           # Streamlit 인터랙티브 대시보드
├── firetrace_fiftyone.py      # FiftyOne 데이터셋 빌더
├── make_firetrace_video.py    # 데모 영상 생성 (광학 흐름 추적)
├── run_firetrace.py           # 전체 파이프라인 실행기
├── reports/                   # 결과 JSON + 시각화 이미지
│   ├── results_combined.json  # 전체 추론 결과 (11장면)
│   ├── origin_*.jpg           # 발원지 시각화 이미지
│   └── temporal_*.jpg         # 시간 진행 스트립 이미지
├── demo/                      # 데모 영상
│   └── firetrace_demo.mp4
├── data/                      # 데이터셋 (별도 다운로드)
├── requirements.txt
└── README.md
```

## 설치

### 요구 사항
- Python 3.12+
- NVIDIA GPU (VRAM 16GB 이상, 2B 모델 기준)
- CUDA 12.4+

### 설치 방법

```bash
cd firetrace

# 가상환경 생성
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux

# PyTorch 설치 (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 의존성 설치
pip install -r requirements.txt
```

### 데이터 준비

`data/` 폴더에 화재 데이터셋을 배치합니다:
```
data/
├── fire_dataset/              # AIHub 화재 감지 데이터셋
│   └── Sample/
│       ├── 01.원천데이터/화재현상/
│       │   ├── 불꽃/0087/JPG/   # 360 프레임
│       │   ├── 연기/0096/JPG/   # 360 프레임
│       │   └── 정상/0077/JPG/   # 360 프레임
│       └── 02.라벨링데이터/
└── fire_videos/               # 추가 화재 영상 (MP4)
```

## 실행

### 전체 파이프라인

```bash
python run_firetrace.py
```

순서대로 실행됩니다:
1. **발원지 시각화** — 대표 프레임에 발원지 마커 + 확산 화살표 표시
2. **대시보드 생성** — Chart.js 기반 HTML 대시보드
3. **FiftyOne 빌드** — 인터랙티브 데이터셋 탐색용

### 개별 실행

```bash
# 1단계: Cosmos-Reason2 추론 (GPU 필요, ~5분/장면)
python fire_detection.py

# 2단계: 발원지 시각화 이미지 생성
python visualize_origin.py

# 3단계: HTML 대시보드 생성
python firetrace_dashboard.py

# 4단계: 데모 영상 생성 (광학 흐름 추적 포함)
python make_firetrace_video.py

# 5단계: Streamlit 대시보드 실행
streamlit run firetrace_app.py --server.headless true

# 6단계: FiftyOne 데이터셋 빌드 + 앱 실행
python firetrace_fiftyone.py --launch
```

## 결과 확인

### 발원지 시각화 이미지
`reports/origin_*.jpg` — 원본 감시 프레임 위에:
- **빨간 십자선** = 예측된 화재 발원지
- **노란 화살표** = 화재/연기 확산 방향
- **정보 패널** = 장면 메타데이터 및 모델 예측

### 시간 진행 스트립
`reports/temporal_*.jpg` — 시간 순서로 배열된 프레임 시퀀스 (화재 진행 과정)

### 데모 영상
`demo/firetrace_demo.mp4` — Lucas-Kanade 광학 흐름으로 발원지를 프레임 간 추적하며 화재 확산을 시각화한 영상

### HTML 대시보드
```bash
python firetrace_dashboard.py
# reports/firetrace_dashboard.html 을 브라우저에서 열기
```

### Streamlit 대시보드
```bash
streamlit run firetrace_app.py --server.headless true
# http://localhost:8501 접속
```

## 작동 원리

1. **영상 전처리**: 감시 카메라 프레임 시퀀스를 시간적 영상으로 변환 (OpenCV). ffmpeg으로 30초 이하로 클리핑, 640px 너비로 리사이즈.

2. **물리 인식 프롬프트**: 화재 안전 전문가 역할로 프롬프트를 구성. 연소 물리학, 대류 패턴, 시간적 화재 역학 분석을 요구.

3. **좌표 기반 발원지 추론**: 모델이 정규화된 (x, y) 좌표로 발원지를 출력하고, 확산 방향 화살표와 함께 원본 프레임 위에 시각적 오버레이.

4. **광학 흐름 추적**: Lucas-Kanade 알고리즘으로 카메라 움직임에 따른 발원지 좌표를 프레임마다 추적 보정하여 데모 영상에 안정적으로 표시.

5. **사고의 연쇄 (Chain-of-Thought)**: 추론 모드를 활성화하여 모델이 물리적 분석 과정을 단계별로 설명. 발원지 추론의 해석 가능한 근거를 제공.

## 기술 참고

- **PyAV 백엔드**: Windows에서 `torchcodec`이 FFmpeg DLL 부재로 동작하지 않아, `transformers.BaseVideoProcessor.fetch_videos`를 `pyav` 기반으로 몽키패치.
- **유니코드 경로 처리**: 한국어 디렉토리명 대응을 위해 `np.fromfile` + `cv2.imdecode` 사용 (`cv2.imread` 대체).
- **좌표 폴백**: 모델이 좌표를 출력하지 않을 경우, 텍스트 기반 발원지 설명을 파싱하여 대략적 위치를 추정.
- **추론 설정**: fps=1 샘플링, temperature=0.6, top_k=20, max_tokens=1024, Chain-of-Thought 추론 활성화.

## 라이선스

NVIDIA Cosmos Cookoff 2026 대회 출품작.
NVIDIA Cosmos-Reason2 모델은 NVIDIA Open Model License에 따라 사용.
