# PP-OCRv6 Python 데모

`demo-ocr.py` 는 DEEPX NPU 로 PP-OCRv6 를 돌리는 PySide6 카메라 데모입니다. 같은 모델을 쓰는 C++ 구현이 `../cpp` 에 있고, 동작 기준은 그쪽입니다.

파이프라인은 **검출 + 인식** 두 단계입니다. v6 에서는 텍스트라인 방향 분류(`textline_ori`), 문서 방향 보정(`doc_ori`), 문서 평탄화(`UVDoc`)를 사용하지 않습니다.

## 실행

저장소 루트에서:

```bash
DX_BACKEND=python ./scripts/run_ocr.sh
```

`scripts/run_ocr.sh` 가 `config.sh` 의 `DX_CAMERA_IDX` 를 `--camera` 로 넘기고, 루트 `.venv` 를 활성화한 뒤 이 스크립트를 실행합니다. `DX_BACKEND` 를 지정하지 않으면 C++ 백엔드(`../cpp/build/cam_ppocr_v6_demo`)가 실행됩니다.

직접 실행할 수도 있습니다.

```bash
source ../../../.venv/bin/activate
python demo-ocr.py --camera 0 --language ch
```

| 인자 | 설명 |
|------|------|
| `--camera N` | 카메라 인덱스 (기본 `0` = `/dev/video0`) |
| `-v, --video PATH` | 카메라 대신 동영상 파일 입력 |
| `--language {ch,korean,german}` | 오버레이 폰트와 인식 모델 선택 (기본 `ch`) |
| `--hide-preview` | 미리보기 패널 숨김 |

## 사전 요구 사항

1. **DXRT 와 Python 바인딩** — 루트 `setup_env.sh` 가 `libdxrt-bin` 에 동봉된 `dx_engine` 휠을 `.venv` 에 설치합니다.

   ```bash
   ./setup_env.sh
   ```

2. **Qt 플랫폼 라이브러리** — PySide6 6.5 이상은 `libxcb-cursor0` 을 요구합니다. 없으면 `Could not load the Qt platform plugin "xcb"` 로 종료합니다.

   ```bash
   sudo apt install -y libxcb-cursor0
   ```

3. **모델 에셋** — `workspace/models/ocr/v6/` 에 있어야 합니다. 없으면 루트에서 `./setup_assets.sh` 를 실행하세요.

   ```text
   workspace/models/ocr/v6/
   ├── det_v6_m_640.dxnn, det_v6_m_960.dxnn
   ├── rec_fixed_v6_ratio_{1,3,5,10,15,25,40}.dxnn
   ├── ppocrv6_dict.txt          # 18,708 행 (+ blank + space = 18,710 클래스)
   └── fonts/                    # 오버레이용 TTF
   ```

## 동작 방식

**전처리는 모델에 fuse 되어 있습니다.** v6 `.dxnn` 의 입력은 `uint8`, NHWC, **BGR**이며 정규화가 없습니다. `engine/preprocessing` 의 `parse_npu_preprocessing_ops` 가 `div`/`normalize` 를 제거하는 이유가 이것입니다. 여기에 `/255` 나 mean/std 를 되살리면 입력이 깨집니다.

**검출** — `det_router` 가 입력 크기로 640/960 모델을 고릅니다. 후처리는 DBNet 이고 상수는 C++ 레퍼런스(`../cpp/ocr_engine.cpp:22-53`)와 맞춰져 있습니다: `thresh=0.7`, `box_thresh=0.6`, `unclip_ratio=1.4`, `max_candidates=50`.

**인식** — 종횡비별로 7개 모델을 상주시킵니다. 버킷 크기는 하드코딩하지 않고 **모델에서 직접 읽습니다**(`RecognitionNode._model_input_hw`). v6 의 ratio 3 모델은 48×144 가 아니라 48×120 이라서, 값을 적어두면 어긋납니다.

| ratio | 입력 (H×W) | 담당 W/H |
|-------|-----------|----------|
| 1 | 48 × 48 | ≤ 1.0 |
| 3 | 48 × 120 | ≤ 2.5 |
| 5 | 48 × 240 | ≤ 5 |
| 10 | 48 × 480 | ≤ 10 |
| 15 | 48 × 720 | ≤ 15 |
| 25 | 48 × 1200 | ≤ 25 |
| 40 | 48 × 1920 | 그 이상 |

크롭은 원근 변환으로 정렬하고, 세로로 긴 크롭(`h > 2w`)은 90° 회전 후 인식합니다. 디코딩은 greedy CTC(인덱스 0 = blank, 반복 축약)이고 점수 0.5 미만은 버립니다.

## 알려진 제약

- C++ 에 있는 **기울기 필터**(`|angle| ≥ 30°` 박스 제거)는 이식하지 않았습니다. 세로쓰기 박스를 잘못 걸러낼 수 있어 검증 전까지 보류합니다.
- `scripts/` 아래 `dxnn_benchmark.py`, `ocr_engine.py` 는 아직 **v5** 기준이고 더 이상 존재하지 않는 `workspace/models/ocr/{server,mobile}` 을 참조합니다. 데모 실행 경로와는 무관합니다.
