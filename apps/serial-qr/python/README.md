# serial-qr — Python 백엔드

이 데모에는 Python 백엔드가 없습니다.

OCR 추론은 `apps/paddle-ocr/cpp/ocr_engine.cpp` (PP-OCRv6, DX-M1 NPU) 를 그대로
재사용하는 C++ HTTP 서버(`../cpp/serial_ocr_server.cpp`)가 담당하고,
화면은 `../web` 의 Vite + React 앱이 담당합니다.

디렉터리는 다른 앱들과 모양을 맞추기 위해서만 남겨둡니다
(`apps/paddle-ocr-web/cpp/README.md` 와 같은 이유).
