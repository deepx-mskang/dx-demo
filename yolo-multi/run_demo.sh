#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

source "${SCRIPT_DIR}/scripts/color_env.sh"
source "${SCRIPT_DIR}/scripts/common_util.sh"

pushd "$SCRIPT_DIR" > /dev/null

# Run install.sh if binary not yet present
if [ ! -f "./bin/yolo_multi_demo" ]; then
    print_colored "yolo_multi_demo not found. Running install.sh..." "INFO"
    bash "${SCRIPT_DIR}/install.sh" || { print_colored "install.sh failed." "ERROR"; exit 1; }
fi

ASSETS="${SCRIPT_DIR}/assets"
if [ ! -d "${ASSETS}/models" ] || [ ! -d "${ASSETS}/videos" ]; then
    print_colored "Assets not found. Running setup.sh..." "INFO"
    bash "${SCRIPT_DIR}/setup.sh" || { print_colored "setup.sh failed." "ERROR"; exit 1; }
fi

EXAMPLE="${SCRIPT_DIR}/config"

BIN="${SCRIPT_DIR}/bin/yolo_multi_demo"

print_colored "Press ESC or Q to stop the demo." "INFO"

echo "0: Multi-Channel Object Detection (YOLOv5)"
echo "1: Multi-Channel Object Detection With PPU (YOLOv5-512)"

prompt="Which demo do you want to run? (default:0): "
printf "%s" "$prompt"

for ((i=20; i>0; i--)); do
    read -t 0.1 -n 1 input 2>/dev/null
    if [ $? -eq 0 ]; then
        read -r rest_input
        select="$input$rest_input"
        break
    fi
    printf "\r%s(%ds) \033[K" "$prompt" "$i"
    sleep 0.9
done

if [ -z "$select" ]; then
    printf "\r%s(timeout) \033[K\n" "$prompt"
    select=0
    echo "Using default: 0"
fi

case $select in
    0) "$BIN" -c "${EXAMPLE}/yolo_multi_demo.json";;
    1) "$BIN" -c "${EXAMPLE}/ppu_yolo_multi_demo.json";;
    *) print_colored "Invalid selection: $select" "ERROR"; exit 1;;
esac

popd > /dev/null
