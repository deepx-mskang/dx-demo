#!/bin/bash 
set -e 

REPO_ROOT=$(cd "$(dirname "$0")" && pwd) 
clean_args=() 
lang_cpp=true 
lang_python=true 
setup_ocr_web=true 
ocr_web_failed=false 

while (( $# )); do 
   case "$1" in 
       --clean) clean_args=(--clean); shift;; 
       --lang) 
           if [ "$2" == "cpp" ]; then 
               lang_cpp=true 
               lang_python=false 
           elif [ "$2" == "python" ]; then 
               lang_cpp=false 
               lang_python=true 
           elif [ "$2" == "all" ]; then 
               lang_cpp=true 
               lang_python=true 
           else 
               echo "Unknown lang: $2. Use cpp, python, or all." 
               exit 1 
           fi 
           shift 2 
           ;; 
       --no-ocr-web) setup_ocr_web=false; shift;; 
       *) 
           echo "Unknown argument: $1" 
           echo "Usage: $0 [--clean] [--lang cpp|python|all] [--no-ocr-web]" 
           echo "  --no-ocr-web   Skip the OCR Web environment setup (heavy: clones repos," 
           echo "                 creates two venvs, installs paddlepaddle)" 
           exit 1 
           ;; 
   esac 
done 

# apps/paddle-ocr-web is vendored from an upstream demo repo (not ours to fork), 
# so this stays here instead of patching apps/paddle-ocr-web/python/build.sh. 
# Its own setup_npu() can build dx_engine from source via cmake against a local 
# dx_rt checkout; that build does not link against libdxrt at all, so the result 
# fails to import with "undefined symbol: dxrt_engine_get_bitmatch_mask". Repair 
# it here using the same prebuilt, self-contained wheel setup_env.sh installs 
# into the top-level .venv (ships with the libdxrt-bin deb package). 
fix_ocr_web_dx_engine() { 
   local fastapi_venv="$REPO_ROOT/apps/paddle-ocr-web/python/PaddleOCR-deepx/deploy/fastapi/venv" 
   [ -x "${fastapi_venv}/bin/python" ] || return 0 

   if "${fastapi_venv}/bin/python" -c 'import dx_engine.capi._pydxrt' > /dev/null 2>&1; then 
       return 0 
   fi 

   echo "OCR Web's dx_engine failed to import; reinstalling it from a prebuilt wheel ..." 

   local py_tag arch dir glob wheel 
   py_tag="cp$("${fastapi_venv}/bin/python" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')" 
   arch=$(uname -m) 
   local wheel_dirs=( 
       "/usr/share/libdxrt-bin/python" 
       "/usr/local/share/libdxrt-bin/python" 
       "${DXRT_DIR:-$HOME/dx_rt}/python_package" 
   ) 

   wheel="" 
   for glob in "dx_engine-*-${py_tag}-*${arch}.whl" "dx_engine-*-${py_tag}-*.whl"; do 
       for dir in "${wheel_dirs[@]}"; do 
           wheel=$(ls ${dir}/${glob} 2>/dev/null | head -n 1) 
           [ -n "${wheel}" ] && break 2 
       done 
   done 

   if [ -z "${wheel}" ]; then 
       echo "Warning: no prebuilt dx_engine wheel found. Looked in:" >&2 
       printf '  %s\n' "${wheel_dirs[@]}" >&2 
       echo "         Install libdxrt-bin, or set DXRT_DIR to your dx_rt source tree." >&2 
       echo "         OCR Web's NPU inference will not work until this is fixed." >&2 
       return 1 
   fi 

   # --force-reinstall: pip sees the same dx-engine version already installed 
   # (the broken cmake build) and would otherwise skip the install as a no-op. 
   if "${fastapi_venv}/bin/pip" install --force-reinstall "${wheel}"; then 
       echo "Fixed: dx_engine reinstalled from $(basename "${wheel}")." 
   else 
       echo "Warning: failed to reinstall dx_engine from $(basename "${wheel}")." >&2 
       return 1 
   fi 
} 

# apps/paddle-ocr-web is vendored (see fix_ocr_web_dx_engine above for why this 
# lives here instead of patching apps/paddle-ocr-web/python/build.sh). Its 
# clone_if_missing() does a plain `git clone`, which - when git-lfs isn't 
# installed - leaves every LFS-tracked example image (examples/*.png) as a 
# ~130-byte pointer stub instead of the real file, so Gradio's example gallery 
# fails with "PIL.UnidentifiedImageError: cannot identify image file ...png". 
fix_ocr_web_lfs_examples() { 
   local web_dir="$REPO_ROOT/apps/paddle-ocr-web/python/PP-OCRv5_Online_demo-deepx" 
   [ -d "${web_dir}/.git" ] || return 0 
   grep -q "filter=lfs" "${web_dir}/.gitattributes" 2>/dev/null || return 0 

   # A real image starts with a binary signature; a pointer stub starts with 
   # this exact line. `head -c` avoids reading a large real file into memory. 
   if ! grep -rlI "^version https://git-lfs.github.com/spec/v1" "${web_dir}/examples" 2>/dev/null | grep -q .; then 
       return 0 
   fi 

   echo "OCR Web's example images are unfetched git-lfs pointers; pulling the real files ..." 

   if ! command -v git-lfs > /dev/null 2>&1; then 
       echo "Warning: git-lfs is not installed, so OCR Web's example gallery will stay broken." >&2 
       echo "         Install it (apt-get install git-lfs) and re-run this script to fix it." >&2 
       return 1 
   fi 

   if (cd "${web_dir}" && git lfs pull); then 
       echo "Fixed: pulled OCR Web's LFS example files." 
   else 
       echo "Warning: 'git lfs pull' failed in ${web_dir}." >&2 
       return 1 
   fi 
} 

if [ "$lang_python" = true ]; then 
   echo "========================================" 
   echo "Setting up Python Environment" 
   echo "========================================" 
   bash "$REPO_ROOT/setup_env.sh" 
   echo "" 

   if [ "$setup_ocr_web" = true ]; then 
       echo "========================================" 
       echo "Setting up OCR Web Environment" 
       echo "========================================" 
       # Clones two repos and builds its own venvs, so it is network-bound and slow. 
       # A failure here must not abort the C++ build below; it is reported at the end. 
       if bash "$REPO_ROOT/apps/paddle-ocr-web/build.sh" "${clean_args[@]}"; then 
           echo "  OK   apps/paddle-ocr-web" 
           fix_ocr_web_dx_engine || true 
           fix_ocr_web_lfs_examples || true 
       else 
           ocr_web_failed=true 
           echo "FAILED: apps/paddle-ocr-web" >&2 
           echo "        Retry with apps/paddle-ocr-web/python/build.sh" >&2 
       fi 
       echo "" 
   else 
       echo "Skipping the OCR Web environment setup (--no-ocr-web)." 
       echo "" 
   fi 
fi 

if [ "$lang_cpp" = true ]; then 
   echo "========================================" 
   echo "Building C++ Projects" 
   echo "========================================" 
   mapfile -t build_scripts < <( 
       find "$REPO_ROOT/apps" -mindepth 3 -maxdepth 4 -name "build.sh" | grep "/cpp/" | sort 
   ) 

   if [ ${#build_scripts[@]} -eq 0 ]; then 
       echo "No build.sh scripts found for C++ projects in apps/ directory." 
   else 
       passed=() 
       failed=() 

       echo "Found ${#build_scripts[@]} build target(s)." 
       echo "" 

       for script in "${build_scripts[@]}"; do 
           rel="${script#$REPO_ROOT/}" 
           echo "========================================" 
           echo "Building: $rel" 
           echo "========================================" 

           if (cd "$(dirname "$script")" && ./build.sh "${clean_args[@]}"); then 
               passed+=("$rel") 
           else 
               failed+=("$rel") 
               echo "FAILED: $rel" >&2 
           fi 
           echo "" 
       done 

       echo "========================================" 
       echo "Build summary (${#passed[@]}/${#build_scripts[@]} succeeded)" 
       echo "========================================" 

       for target in "${passed[@]}"; do 
           echo "  OK   $target" 
       done 

       if [ ${#failed[@]} -gt 0 ]; then 
           for target in "${failed[@]}"; do 
               echo "  FAIL $target" 
           done 
           if [ "$ocr_web_failed" = true ]; then 
               echo "  FAIL apps/paddle-ocr-web (OCR Web environment)" >&2 
           fi 
           exit 1 
       fi 
   fi 
fi 

if [ "$ocr_web_failed" = true ]; then 
   echo "========================================" >&2 
   echo "The OCR Web environment setup failed - the OCR Web demo will not start." >&2 
   echo "Retry with apps/paddle-ocr-web/python/build.sh" >&2 
   echo "========================================" >&2 
   exit 1 
fi