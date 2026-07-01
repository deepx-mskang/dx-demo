# yolo_multi_demo

`yolo_multi_demo`는 여러 입력 채널을 동시에 처리하는 멀티 채널 YOLO 객체 검출 데모입니다.

## 개요

- 실행 바이너리: `bin/yolo_multi_demo`
- 설정 파일 기반 실행
- 멀티 채널 객체 검출 시나리오 지원

실행 시 아래 설정 중 하나를 선택할 수 있습니다.

- `0`: `config/yolo_multi_demo.json`
- `1`: `config/ppu_yolo_multi_demo.json`

## 실행 방법

```bash
cd yolo_multi_demo
./run_demo.sh
```

## 실행 스크립트 동작

`run_demo.sh`는 다음을 자동으로 처리합니다.

1. `bin/yolo_multi_demo`가 없으면 `install.sh`를 실행합니다.
2. `assets/models` 또는 `assets/videos`가 없으면 `setup.sh`를 실행합니다.
3. 메뉴에서 선택한 설정 파일을 `-c` 옵션으로 전달해 데모를 시작합니다.

입력이 없으면 기본값 `0`이 선택됩니다.

## 종료 방법

- 실행 중 `ESC` 또는 `Q` 키를 누르면 종료할 수 있습니다.

## 스크린샷

**0: Multi Channel Object Detection**

![yolo_multi_demo](img/yolo_multi_demo_screenshot.png)

**1: Multi Channel Object Detection With PPU**

![yolo_multi_demo_with_ppu](img/yolo_multi_demo_with_ppu_screenshot.png)
