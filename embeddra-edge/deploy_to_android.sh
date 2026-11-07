#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "EMBEDDRA EDGE RUNTIME: ANDROID USB DEPLOYMENT AUTOMATION"
echo "=========================================================="

CHECKPOINT_PATH="${1:-../checkpoints/maria_1b_20B_weights.pt}"

# 1. Check ADB
if ! command -v adb &> /dev/null; then
    echo "[ERROR] 'adb' command not found. Please install Android SDK platform-tools."
    exit 1
fi

# 2. Check Device
echo "[1/4] Detecting connected Android devices..."
DEVICE=$(adb devices | grep -w "device" | head -n 1 | awk '{print $1}')
if [ -z "$DEVICE" ]; then
    echo "[ERROR] No authorized Android device detected over USB."
    echo "Please enable USB debugging on your device and accept the authorization prompt."
    exit 1
fi
MODEL=$(adb shell getprop ro.product.model | tr -d '\r')
echo "[OK] Connected Device: $MODEL ($DEVICE)"

# 3. Create target directory & Push weights
TARGET_DIR="/sdcard/Android/data/com.embeddra.edge/files"
echo "[2/4] Ensuring target directory exists on device: $TARGET_DIR"
adb shell "mkdir -p $TARGET_DIR"

if adb shell "[ -f $TARGET_DIR/maria_1b_20B_weights.pt ]"; then
    echo "[INFO] Maria-1B weights already exist on device. Skipping transfer."
else
    echo "[3/4] Pushing Maria-1B weights ($CHECKPOINT_PATH) to phone..."
    adb push "$CHECKPOINT_PATH" "$TARGET_DIR/"
    echo "[OK] Model weights pushed successfully."
fi

# 4. Launch app
echo "[4/4] Starting Embeddra Edge on phone..."
adb shell "am start -n com.embeddra.edge/.MainActivity" 2>/dev/null || true

echo "=========================================================="
echo "DEPLOYMENT COMPLETE! Ready for offline inference."
echo "=========================================================="

