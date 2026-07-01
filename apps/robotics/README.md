# robotics_demo

```
sudo apt install -y libssl-dev openssl pkg-config
```

`robotics_demo`는 얼굴 검출, 얼굴 정렬, 얼굴 식별 모델을 함께 사용하는 얼굴 인식 데모입니다.

## 개요

- 실행 바이너리: `bin/robotics_demo`
- 사용 모델:
  - `assets/hyundai_models/HyundaiPytorchHalfpixel_1.dxnn`
  - `assets/hyundai_models/HyundaiFaceAlignment.dxnn`
  - `assets/hyundai_models/HyundaiFaceID_1.dxnn`
- 샘플 입력:
  - 기준 이미지: `sample/base_image.png`
  - 비교 이미지: `sample/face.png`

이 데모는 여러 얼굴 관련 모델을 순차적으로 사용해 얼굴 위치를 찾고, 정렬한 뒤, 식별 결과를 보여주는 흐름으로 구성되어 있습니다.

## 실행 방법

```bash
cd robotics_demo
./run_demo.sh
```

## 실행 스크립트 동작

`run_demo.sh`는 다음을 자동으로 확인합니다.

1. `bin/robotics_demo`가 없으면 `install.sh`를 실행합니다.
2. `assets/hyundai_models` 디렉터리가 없으면 `setup.sh`를 실행합니다.
3. 준비된 모델 3개와 샘플 이미지 2개를 사용해 데모를 실행합니다.

## 종료 방법

- 실행 중 `ESC` 또는 `Q` 키를 누르면 종료할 수 있습니다.

## 스크린샷

![face_recognition_demo](img/face_recognition_demo_screenshot.png)
