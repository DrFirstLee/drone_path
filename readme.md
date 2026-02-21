# 🚁 드론 비행 경로 분석 도구

CSV 파일로 기록된 드론 비행 데이터를 분석하여  
지도 시각화(HTML)와 세그먼트별 CSV 파일을 자동으로 만들어줍니다.

---

## 📁 폴더 구조

```
drone_path/
├── logic.py            ← 분석 스크립트 (이걸 실행!)
├── input.csv           ← 분석할 드론 비행 데이터
├── requirements.txt    ← 설치해야 할 패키지 목록
├── index.html          ← 웹 UI (선택 사항)
├── logic.html          ← 로직 설명 페이지 (선택 사항)
└── .output/            ← 결과물이 여기에 저장됩니다 (자동 생성)
    ├── flight_path_segmented.html
    ├── full_result.csv
    ├── Line-0.csv
    ├── Rotate-1.csv
    ├── Rotate_Error-3.csv
    └── ...
```

---

## ⚙️ 처음 한 번만 하는 사전 준비

### 1. Python 설치 확인

터미널(명령 프롬프트)을 열고 아래 명령어를 입력해보세요.

```bash
python --version
```

`Python 3.10.x` 처럼 숫자가 나오면 OK입니다.  
아무것도 안 나오면 [python.org](https://www.python.org/downloads/)에서 Python을 먼저 설치해주세요.

---

### 2. 필수 패키지 설치

터미널에서 이 폴더로 이동한 뒤 아래 명령어를 실행하세요.

```bash
pip install -r requirements.txt
```

> 💡 한 번 설치하면 다음부터는 안 해도 됩니다.

---

## 🚀 사용 방법

### 기본 실행 (가장 간단한 방법)

1. 분석할 CSV 파일의 이름을 **`input.csv`** 로 변경해서 이 폴더에 넣으세요.
2. 터미널에서 아래 명령어를 실행하세요.

```bash
python logic.py
```

3. 잠시 기다리면 **`.output/`** 폴더 안에 결과물이 생성됩니다!

---

### 파일 이름을 바꾸기 싫을 때 (파일 직접 지정)

```bash
python logic.py --input 내파일이름.csv
```

예시:
```bash
python logic.py --input flight_20260221.csv
```

---

### 결과 저장 폴더도 바꾸고 싶을 때

```bash
python logic.py --input 내파일이름.csv --output 결과폴더이름
```

예시:
```bash
python logic.py --input flight_20260221.csv --output results/feb21
```

---

## 📋 CSV 파일 형식

CSV 파일은 반드시 아래 **13개 컬럼 순서**를 지켜야 합니다.  
헤더(컬럼명)는 없어도 됩니다.

| 순서 | 컬럼명 | 설명 |
|:---:|---|---|
| 1 | Time | Unix 타임스탬프 (예: 1708500000) |
| 2 | Longitude | 경도 |
| 3 | Latitude | 위도 |
| 4 | Altitude | 고도 (m) |
| 5 | Roll | 롤각 (도°) |
| 6 | Pitch | 피치각 (도°) |
| 7 | Yaw | 요각 (도°) |
| 8 | Mag1_X | 자력계1 X축 |
| 9 | Mag1_Y | 자력계1 Y축 |
| 10 | Mag1_Z | 자력계1 Z축 |
| 11 | Mag2_X | 자력계2 X축 |
| 12 | Mag2_Y | 자력계2 Y축 |
| 13 | Mag2_Z | 자력계2 Z축 |

> ⚠️ 앞에 불필요한 컬럼이 있어도 자동으로 찾아냅니다.  
> (Time 값이 500,000보다 큰 컬럼을 Unix 타임스탬프로 인식)

---

## 📂 결과물 설명

`.output/` 폴더 안에 아래 파일들이 생성됩니다.

| 파일 | 설명 |
|---|---|
| `flight_path_segmented.html` | **지도 시각화** — 브라우저로 열어서 확인 |
| `full_result.csv` | 전체 비행 데이터 + 분석 결과 컬럼 포함 |
| `Line-N.csv` | 직선 비행 구간 N번 데이터 |
| `Rotate-N.csv` | 회전 구간 N번 데이터 |
| `Rotate_Error-N.csv` | 오분류된 회전 구간 N번 (실제로는 직선) |

### 지도에서 색상 의미

| 표시 | 의미 |
|---|---|
| 🔵 실선 | **Line** — 직선 비행 구간 |
| 🟣 점선 (짧은) | **Rotate** — 실제 회전 구간 |
| 🟠 점선 (긴) + ⚠️ | **Rotate_Error** — HMM이 회전으로 오분류한 직선 구간 |
| ⚫ 연한 검정 실선 | 전체 원본 경로 (배경) |

---

## 🔍 분석 로직 요약

1. **HMM 클러스터링** — Yaw, Roll, 속도 피처로 직선(Line) vs 회전(Rotate) 자동 분류
2. **Rotate_Error 판별** — PCA 선형성(≥ 0.97) **AND** Yaw 변화 < 45° 이면 오분류로 판정
3. **체인 병합** — `Line-Error-Line` 패턴을 하나의 직선 세그먼트로 통합

> 자세한 설명은 `logic.html` 파일을 브라우저로 열어 확인하세요.

---

## ❓ 자주 발생하는 문제

### "`python` 명령어를 찾을 수 없습니다"
→ Python이 설치되지 않았거나 PATH 등록이 안 된 것입니다.  
Python을 설치할 때 **"Add Python to PATH"** 옵션을 체크해주세요.

### "`ModuleNotFoundError`가 뜹니다"
→ 패키지 설치가 안 된 것입니다. 아래를 다시 실행해보세요.
```bash
pip install -r requirements.txt
```

### "`Timestamp 컬럼을 찾을 수 없습니다`" 오류
→ CSV 파일의 첫 5개 컬럼 중에 Unix 타임스탬프(500,000 이상 숫자)가 없습니다.  
파일 형식을 다시 확인해주세요.

### "결과가 이상하게 나와요"
→ `hmmlearn`이 설치되지 않으면 K-Means로 대체되어 정확도가 낮아질 수 있습니다.
```bash
pip install hmmlearn
```

---

## 🛠️ 개발 환경

- Python 3.10+
- numpy, pandas, scikit-learn, hmmlearn, folium
