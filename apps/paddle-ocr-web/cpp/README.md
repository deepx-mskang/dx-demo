# paddle-ocr-web / cpp

이 데모에는 **C++ 구현이 없습니다.**

PP-OCRv5 Web 데모는 업스트림에서 Python 으로만 제공됩니다.

- Web UI: [PP-OCRv5_Online_demo-deepx](https://github.com/DEEPX-AI/PP-OCRv5_Online_demo-deepx) — Gradio (`app.py`)
- OCR 서버: [PaddleOCR-deepx](https://github.com/DEEPX-AI/PaddleOCR-deepx) — FastAPI (`deploy/fastapi/ocr_service.py`)

NPU 추론은 서버가 `dx_engine`(DX-RT Python 바인딩)을 통해 수행하므로, C++ 경로 없이도 NPU 가속이 동작합니다.

카메라 실시간 OCR 의 C++ 구현이 필요하면 `apps/paddle-ocr/cpp/` 를 참고하세요.

환경 구성은 `../python/build.sh` (또는 `../build.sh`) 입니다.
