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
4. 클릭 시점에 선택된 언어를 `--language <code>` 인자로 전달하여 스크립트를 `/bin/bash`로 실행합니다.
5. `DX_LAUNCHER_READY_FILE` 환경변수에 ready 파일 경로를 전달합니다.
6. 지정된 대기 시간이 지나거나, 실행된 앱이 ready 파일을 갱신하면 버튼 상태를 복구합니다.

언어 코드는 다음 중 하나입니다.

- `en`: 영어
- `ko`: 한국어
- `zh`: 중국어
- `ja`: 일본어

예를 들어 한국어를 선택한 상태에서 버튼을 누르면 다음과 같은 형태로 실행됩니다.

```bash
/bin/bash scripts/run_demo.sh --language ko
```

### 실행 스크립트에서 언어 적용하기

실행 스크립트는 `--language`를 파싱한 뒤, 동일한 애플리케이션에 전달하거나 언어별 설정을 선택할 수 있습니다.

```bash
#!/usr/bin/env bash
set -euo pipefail

LANGUAGE_CODE="en"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --language)
            if [[ $# -lt 2 ]]; then
                echo "--language requires a value" >&2
                exit 2
            fi
            LANGUAGE_CODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

case "$LANGUAGE_CODE" in
    en|ko|zh|ja) ;;
    *)
        echo "Unsupported language: $LANGUAGE_CODE" >&2
        exit 2
        ;;
esac

exec python3 demo.py --language "$LANGUAGE_CODE"
```

언어별로 완전히 다른 스크립트가 필요하다면 같은 파싱 코드 뒤에서 분기할 수 있습니다.

```bash
case "$LANGUAGE_CODE" in
    en) exec ./run_demo_en.sh ;;
    ko) exec ./run_demo_ko.sh ;;
    zh) exec ./run_demo_zh.sh ;;
    ja) exec ./run_demo_ja.sh ;;
esac
```

Python 애플리케이션에서는 다음처럼 받을 수 있습니다.

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--language", choices=["en", "ko", "zh", "ja"], default="en")
args = parser.parse_args()
print(f"selected language: {args.language}")
```

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
