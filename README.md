# 🛡️ AI Guard System
### AI를 활용한 군 경계 강화 시스템

---

## 📖 프로젝트 소개

AI Guard System은 YOLO 기반 객체 탐지 기술을 활용하여 군 경계 지역의 CCTV 영상을 실시간으로 분석하고, 제한구역 침입 여부를 자동으로 감지하는 AI 기반 경계 지원 시스템입니다.

기존의 군 경계 시스템은 감시병이 장시간 CCTV를 직접 모니터링해야 하므로 집중력 저하와 피로 누적으로 인해 침입 상황을 놓칠 가능성이 존재합니다.

본 프로젝트는 AI 객체 탐지와 객체 추적(Object Tracking)을 활용하여 이러한 문제를 보완하고, 경계 인원의 부담을 줄이며 신속한 상황 인지를 지원하는 것을 목표로 합니다.

---

# 🎯 개발 목표

- 실시간 CCTV 영상 분석
- YOLO 기반 사람 객체 탐지
- 객체별 고유 ID 부여 및 추적
- 제한구역 침입 감지
- 실시간 웹 모니터링 제공
- 침입 통계 시각화
- 군 경계 업무 효율 향상

---

# 💡 주요 기능

## 1. 실시간 객체 탐지

YOLO 모델을 이용하여 영상 속 사람을 실시간으로 탐지합니다.

- Person Detection
- Bounding Box 표시
- Confidence Score 표시

---

## 2. 객체 추적(Object Tracking)

탐지된 사람에게 고유 ID를 부여하여 이동 경로를 지속적으로 추적합니다.

예시

```
Person #1
Person #2
Person #3
```

동일 인물은 화면에서 이동하더라도 동일한 ID를 유지합니다.

---

## 3. 제한구역 침입 감지

미리 설정된 제한구역(Restricted Area)에 사람이 진입하면 자동으로 침입을 감지합니다.

감지 내용

- 제한구역 진입
- 체류 시간 계산
- 위험 여부 표시

---

## 4. 웹 기반 모니터링

Flask를 이용하여 AI 분석 결과를 웹페이지에서 실시간으로 확인할 수 있습니다.

웹 화면에서는

- CCTV 영상
- 객체 정보
- 제한구역 상태
- 탐지 현황

등을 확인할 수 있습니다.

---

## 5. 통계 그래프

웹페이지에서 실시간 탐지 정보를 기반으로 통계를 제공합니다.

예시

- 현재 탐지 인원
- 제한구역 침입 횟수
- 시간대별 탐지 수

---

# 🖥️ 개발 환경

| 항목 | 내용 |
|------|------|
| Language | Python 3 |
| IDE | Visual Studio Code |
| Framework | Flask |
| AI Model | YOLO |
| Computer Vision | OpenCV |
| Front-End | HTML / CSS / JavaScript |
| Version Control | Git / GitHub |

---

# 📂 프로젝트 구조

```
AI-Guard-System
│
├── main.py
├── app.py
├── requirements.txt
├── README.md
│
├── templates
│   └── index.html
│
├── static
│   ├── css
│   │      style.css
│   ├── js
│   │      script.js
│   └── images
│
├── models
│   └── yolov8n.pt
│
├── videos
│
└── utils
```

※ 실제 프로젝트 구조에 따라 일부 폴더명은 달라질 수 있습니다.

---

# ⚙️ 설치 방법

## 1. 프로젝트 다운로드

```bash
git clone https://github.com/boramae673/AI-Guard-System.git
```

프로젝트 폴더 이동

```bash
cd AI-Guard-System
```

---

## 2. 가상환경 생성

```bash
python -m venv .venv
```

---

## 3. 가상환경 실행

Windows

```bash
.venv\Scripts\activate
```

Mac / Linux

```bash
source .venv/bin/activate
```

---

## 4. 라이브러리 설치

```bash
pip install -r requirements.txt
```

---

## 5. 실행

프로젝트에 따라

```bash
python main.py
```

또는

```bash
python app.py
```

---

## 6. 웹 접속

브라우저에서

```
http://127.0.0.1:5000
```

접속합니다.

---

# 📸 시스템 동작 과정

1. CCTV 영상 입력

↓

2. YOLO 객체 탐지

↓

3. 객체 추적(ID 부여)

↓

4. 제한구역 침입 여부 판단

↓

5. 웹페이지에 실시간 표시

↓

6. 탐지 결과 및 통계 저장

---

# 🚀 기대 효과

- 군 경계 업무 자동화
- 감시병 피로도 감소
- 침입 상황 조기 발견
- 실시간 상황 인지
- AI 기반 스마트 경계 시스템 구축

---

# 🔮 향후 발전 방향

- 얼굴 인식 기능 추가
- 군인/민간인 자동 구분
- 위험 행동 분석
- 드론 영상 연동
- 알림 시스템 구축
- 데이터베이스 저장 기능
- 모바일 모니터링 지원

---

# 👥 Team

강원대학교 디지털밀리터리학과

**AI를 활용한 군 경계 강화 시스템**

팀원

- 배서윤
- 김다교
- 노주헌

---

# 📄 License

본 프로젝트는 교육 및 학습 목적의 프로젝트입니다.

---

# 📬 Contact

강원대학교 디지털밀리터리학과

AI Guard System Project