<#
.SYNOPSIS
    Embeddra Edge Runtime - One-Click USB Android Deployment Script
.DESCRIPTION
    Automates device detection, checkpoint transfer, permissions, and app deployment.
#>

param(
    [string]$CheckpointPath = "..\checkpoints\maria_1b_20B_weights.pt"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "EMBEDDRA EDGE RUNTIME: ANDROID USB DEPLOYMENT AUTOMATION" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Resolve ADB location automatically
$adbBin = $null
$candidatePaths = @(
    "adb",
    "$PSScriptRoot\..\platform-tools\adb.exe",
    "$PSScriptRoot\platform-tools\adb.exe",
    "C:\platform-tools\adb.exe",
    "$env:USERPROFILE\Downloads\platform-tools\adb.exe",
    "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
)

foreach ($candidate in $candidatePaths) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $adbBin = $candidate
        break
    } elseif (Test-Path $candidate) {
        $adbBin = $candidate
        break
    }
}

if (-not $adbBin) {
    Write-Host "[ERROR] 'adb.exe' was not found." -ForegroundColor Red
    Write-Host "Please extract 'platform-tools' into this folder or install Android SDK." -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Using ADB: $adbBin" -ForegroundColor Green
function adb { & $adbBin @args }

# 2. Check for connected devices
Write-Host "`n[1/5] Detecting connected Android devices..." -ForegroundColor Yellow
$devicesOutput = adb devices
$connectedDevices = $devicesOutput | Select-String -Pattern "\bdevice\b"

if (-not $connectedDevices) {
    Write-Host "[ERROR] No connected Android device detected." -ForegroundColor Red
    Write-Host "1. Ensure your phone is connected via USB." -ForegroundColor White
    Write-Host "2. Enable USB Debugging in Developer Options on your phone." -ForegroundColor White
    Write-Host "3. Unlock your phone and accept the 'Allow USB debugging' prompt." -ForegroundColor White
    exit 1
}

$deviceInfo = adb shell getprop ro.product.model
Write-Host "[OK] Connected Device Detected: $deviceInfo" -ForegroundColor Green

# 3. Check source checkpoint
if (-not (Test-Path $CheckpointPath)) {
    # Try resolving relative to script root
    $altPath = Join-Path $PSScriptRoot "..\checkpoints\maria_1b_20B_weights.pt"
    if (Test-Path $altPath) {
        $CheckpointPath = $altPath
    } else {
        Write-Host "[ERROR] Checkpoint file not found at: $CheckpointPath" -ForegroundColor Red
        exit 1
    }
}

$fileSizeMb = [math]::Round((Get-Item $CheckpointPath).Length / 1MB, 2)
Write-Host "[OK] Source Checkpoint: $CheckpointPath ($fileSizeMb MB)" -ForegroundColor Green

# 4. Prepare target directory on phone
$targetDir = "/sdcard/Android/data/com.embeddra.edge/files"
Write-Host "`n[2/5] Creating application storage directory on device..." -ForegroundColor Yellow
adb shell "mkdir -p $targetDir"

# 5. Check if file already exists on phone
$existingCheck = adb shell "ls -l $targetDir/maria_1b_20B_weights.pt" 2>$null
if ($existingCheck -match "maria_1b_20B_weights.pt") {
    Write-Host "[INFO] Model weights already exist on device. Skipping transfer." -ForegroundColor Cyan
} else {
    Write-Host "`n[3/5] Pushing Maria-1B weights to device ($fileSizeMb MB)... This may take ~30-60s over USB." -ForegroundColor Yellow
    adb push $CheckpointPath "$targetDir/"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Weights transferred successfully!" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to push weights to phone storage." -ForegroundColor Red
        exit 1
    }
}

# 6. Build and install Android APK
Write-Host "`n[4/5] Building & Installing Android APK..." -ForegroundColor Yellow
$gradlew = Join-Path $PSScriptRoot "android\gradlew.bat"
if (Test-Path $gradlew) {
    Push-Location (Join-Path $PSScriptRoot "android")
    .\gradlew.bat assembleRelease
    Pop-Location

    $apkPath = Join-Path $PSScriptRoot "android\app\build\outputs\apk\release\app-release.apk"
    if (Test-Path $apkPath) {
        Write-Host "Installing APK to device..." -ForegroundColor Yellow
        adb install -r $apkPath
        Write-Host "[OK] Embeddra Edge APK installed successfully!" -ForegroundColor Green
    }
} else {
    Write-Host "[NOTE] Gradlew not found in local path. If using Android Studio, press Run to install APK." -ForegroundColor Cyan
}

# 7. Launch application
Write-Host "`n[5/5] Launching Embeddra Edge on device..." -ForegroundColor Yellow
adb shell "am start -n com.embeddra.edge/.MainActivity" 2>$null

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE! Ready for offline inference." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

