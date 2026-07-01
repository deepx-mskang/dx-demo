# Tensor Porting Harness

이 폴더는 C++ ↔ Python 혹은 다른 환경 간의 AI 모델 포팅 시 전/후처리 메모리가 정확히 일치하는지(Parity Check) 검증하기 위한 도구 모음입니다. 

프로젝트에 구애받지 않고 어느 곳이든 이 폴더(`tensor_harness`) 전체를 복사해서 사용할 수 있도록 독립적으로 구성되었습니다.

## 구성 파일
1. `tensor_harness.hpp`: C++ 프로젝트에 include 하여 사용할 수 있는 덤프 유틸리티.
2. `tensor_harness.py`: Python 프로젝트에서 import 하여 사용할 수 있는 덤프 유틸리티.
3. `compare_tensors.py`: 양쪽에서 뽑아낸 `.bin` 파일 간의 오차를 검증하는 CLI 툴.

## 사용 방법

### 1. C++ 코드에서 텐서 덤프하기
코드 최상단에 헤더를 포함하고, NPU에 전달하기 직전의 `std::vector`나 배열 포인터를 덤프합니다.
```cpp
#include "tensor_harness.hpp"

// ... (전처리 완료 후)
dx_harness::dump_tensor(my_input_vector, "input_cpp.bin");
```

### 2. Python 코드에서 텐서 덤프하기
Python 파일에서 import 후 numpy 배열을 덤프합니다.
```python
import tensor_harness

# ... (전처리 완료 후)
tensor_harness.dump_tensor(my_input_numpy, "input_py.bin")
```

### 3. 검증하기 (Terminal)
터미널에서 제공된 `compare_tensors.py` 툴을 사용하여 두 파일 간의 오차를 비교합니다.
```bash
python3 compare_tensors.py input_py.bin input_cpp.bin
```

결과가 `PASS` (Max Diff가 1e-6 이하) 라면 전처리가 양쪽 언어에서 완벽하게 동일하게 구현되었음을 의미합니다.
