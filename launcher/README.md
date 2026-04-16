# Launcher

PyQt5 기반 DEEPX demo launcher입니다. `launcher/main.py`에서 카드 UI를 만들고, 각 버튼이 `../scripts/` 아래 실행 스크립트를 호출합니다.

## Directory Layout

현재 런처는 아래 구조를 전제로 동작합니다.

```text
dx-demos/
├── launcher/
│   ├── main.py
│   ├── assets/
│   └── ready/
└── scripts/
    ├── run_launcher.sh
    ├── run_modelzoo.sh
    └── ...
```

- 이미지 경로는 `launcher/main.py` 기준 상대경로로 해석됩니다.
- 스크립트 경로도 `launcher/main.py` 기준 상대경로로 해석됩니다.
  예: `../scripts/run_modelzoo.sh`
- `launcher/ready/`는 버튼 클릭 후 대기 상태를 추적하는 임시 ready 파일 저장 위치이며, 없으면 자동 생성됩니다.

## Requirements

- Python 3
- `PyQt5`

예시:

```bash
sudo apt update
sudo apt install -y python3-pyqt5
```

## Run

저장소 루트에서 실행:

```bash
python3 launcher/main.py
```

또는 런처 디렉터리에서 실행:

```bash
cd launcher
python3 main.py
```

상대 스크립트 경로는 `main.py` 위치를 기준으로 해석되므로, 현재 작업 디렉터리가 어디인지에 영향받지 않습니다.

## Configuration

런처의 주요 설정은 모두 `launcher/main.py` 상단에 있습니다.

- `NUM_ITEMS`: 화면에 표시할 카드 수
- `GRID_COLUMNS`: 카드 그리드 열 수
- `WINDOW_WIDTH`, `WINDOW_HEIGHT`: 런처 창 크기
- `LAUNCHER_ITEMS`: 카드 제목, 이미지, 실행 스크립트, 버튼 라벨, 대기 시간 설정

각 item 예시는 아래와 같습니다.

```python
{
    "title": "DEEPX Model Zoo",
    "image": "assets/demo-modelzoo.png",
    "video_label": "Open",
    "camera_label": "Close",
    "video_script": "../scripts/run_modelzoo.sh",
    "camera_script": "../scripts/kill_modelzoo.sh",
}
```

지원하는 주요 키:

- `title`: 카드 제목
- `image`: 카드 이미지 경로
- `video_script`, `camera_script`: 기본 2개 버튼에 연결할 스크립트
- `video_label`, `camera_label`: 버튼 텍스트 override
- `loading_sec`: 카드 전체 기본 대기 시간
- `video_loading_sec`, `camera_loading_sec`: 버튼별 대기 시간 override
- `extra_buttons`: 추가 버튼 목록

`extra_buttons` 예시:

```python
"extra_buttons": [
    {"label": "Export", "script": "../scripts/export.sh", "loading_sec": 15},
]
```

## Button Behavior

버튼을 누르면 런처는 다음 순서로 동작합니다.

1. 대상 스크립트가 존재하는지 확인합니다.
2. 모든 버튼을 일시적으로 비활성화하고, 클릭한 버튼 텍스트를 `Wait`로 바꿉니다.
3. `launcher/ready/` 아래에 고유한 ready 파일을 생성합니다.
4. 스크립트를 `/bin/bash`로 실행하면서 `DX_LAUNCHER_READY_FILE` 환경변수에 ready 파일 경로를 전달합니다.
5. 지정된 대기 시간이 지나거나, 실행된 앱이 ready 파일을 갱신하면 버튼 상태를 복구합니다.

즉, 실행 스크립트나 하위 앱에서 준비 완료 시점을 앞당기고 싶다면 `DX_LAUNCHER_READY_FILE`을 사용하면 됩니다.

예시:

```bash
touch "$DX_LAUNCHER_READY_FILE"
```

또는

```bash
echo ready > "$DX_LAUNCHER_READY_FILE"
```

## Notes

- 스크립트는 `subprocess.Popen(..., start_new_session=True)`로 실행됩니다.
- 스크립트의 작업 디렉터리는 해당 스크립트가 있는 폴더로 설정됩니다.
- 이미지 파일이 없으면 카드 영역에 `No image`가 표시됩니다.
