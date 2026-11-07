# Embeddra Edge Runtime - Portable Android APK Builder & Deployer
# Automatically packages Embeddra Edge APK and deploys to connected Samsung phone without Android Studio.

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  EMBEDDRA EDGE: PORTABLE ANDROID APK BUILDER & DEPLOYER" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Setup Java Environment
$JdkPath = "$ProjectRoot\jdk\jdk-17.0.10+7"
if (-not (Test-Path "$JdkPath\bin\java.exe")) {
    Write-Host "[1/5] Downloading Portable OpenJDK 17..." -ForegroundColor Yellow
    $JdkZip = "$ProjectRoot\jdk.zip"
    Invoke-WebRequest -Uri "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.10%2B7/OpenJDK17U-jdk_x64_windows_hotspot_17.0.10_7.zip" -OutFile $JdkZip
    Expand-Archive -Path $JdkZip -DestinationPath "$ProjectRoot\jdk" -Force
    Remove-Item $JdkZip
}
$env:JAVA_HOME = $JdkPath
$env:Path = "$JdkPath\bin;$env:Path"
$env:_JAVA_OPTIONS = "-Djava.net.preferIPv4Stack=true"
Write-Host "[1/5] Java 17 Ready: OpenJDK 17.0.10 LTS at $JdkPath" -ForegroundColor Green

# 2. Setup Portable Gradle
$GradleDir = "$ScriptDir\tools\gradle-8.7"
if (-not (Test-Path "$GradleDir\bin\gradle.bat")) {
    Write-Host "[2/5] Downloading Portable Gradle 8.7 (120 MB)..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "$ScriptDir\tools" | Out-Null
    $GradleZip = "$ScriptDir\tools\gradle.zip"
    Invoke-WebRequest -Uri "https://services.gradle.org/distributions/gradle-8.7-bin.zip" -OutFile $GradleZip
    Expand-Archive -Path $GradleZip -DestinationPath "$ScriptDir\tools" -Force
    Remove-Item $GradleZip
}
$GradleBin = "$GradleDir\bin\gradle.bat"
Write-Host "[2/5] Gradle 8.7 Ready." -ForegroundColor Green

# 3. Setup Android Commandline Tools (SDK)
$SdkDir = "$ProjectRoot\android-sdk"
if (-not (Test-Path "$SdkDir\cmdline-tools\latest\bin\sdkmanager.bat")) {
    Write-Host "[3/5] Setting up minimal Android SDK Commandline Tools..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "$SdkDir" | Out-Null
    $CmdZip = "$ScriptDir\tools\cmdline-tools.zip"
    Invoke-WebRequest -Uri "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip" -OutFile $CmdZip
    Expand-Archive -Path $CmdZip -DestinationPath "$SdkDir\temp_tools" -Force
    New-Item -ItemType Directory -Force -Path "$SdkDir\cmdline-tools\latest" | Out-Null
    Copy-Item -Path "$SdkDir\temp_tools\cmdline-tools\*" -Destination "$SdkDir\cmdline-tools\latest" -Recurse -Force
    Remove-Item -Recurse -Force "$SdkDir\temp_tools"
    Remove-Item $CmdZip
}
$env:ANDROID_HOME = $SdkDir
$env:ANDROID_SDK_ROOT = $SdkDir
Write-Host "[3/5] Android SDK Path Configured: $SdkDir" -ForegroundColor Green

# Accept licenses and install platform 34 and cmake
$SdkManager = "$SdkDir\cmdline-tools\latest\bin\sdkmanager.bat"
if (-not (Test-Path "$SdkDir\platforms\android-34")) {
    Write-Host "Installing Android Platform 34 & Build Tools..." -ForegroundColor Yellow
    cmd.exe /c "echo y | `"$SdkManager`" --licenses" | Out-Null
    & $SdkManager "platforms;android-34" "build-tools;34.0.0" "cmake;3.22.1" "ndk;26.1.10909125"
}

# 4. Compile APK
Write-Host "[4/5] Building Embeddra Edge APK (assembleDebug)..." -ForegroundColor Cyan
Set-Location $ScriptDir
$ApkPath = "$ScriptDir\app\build\outputs\apk\debug\app-debug.apk"
Remove-Item -Force $ApkPath -ErrorAction SilentlyContinue
& $GradleBin assembleDebug --no-daemon --stacktrace
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $ApkPath)) {
    Write-Host "Build failed: APK not found at $ApkPath" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] APK compiled successfully: $ApkPath" -ForegroundColor Green

# 5. Deploy to Phone
$AdbBin = "$ProjectRoot\platform-tools\adb.exe"
Write-Host "[5/5] Deploying APK directly to connected Samsung phone..." -ForegroundColor Cyan
& $AdbBin -s RZCX61AJBKX install -r $ApkPath
& $AdbBin -s RZCX61AJBKX reverse tcp:8080 tcp:8080
Write-Host "Launching Embeddra Edge on phone screen..." -ForegroundColor Green
& $AdbBin -s RZCX61AJBKX shell "am start -n com.embeddra.edge/.MainActivity"

Write-Host "=================================================================" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE! Check your Samsung phone screen." -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green


