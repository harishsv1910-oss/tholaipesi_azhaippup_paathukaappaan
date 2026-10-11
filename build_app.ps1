$ErrorActionPreference = "Stop"

Write-Host "Building FraudCallScanner app..."
py -3.13 -m PyInstaller --clean --noconfirm FraudCallScanner.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

$appDir = Join-Path $PSScriptRoot "dist\FraudCallScanner"
New-Item -ItemType Directory -Force -Path (Join-Path $appDir "Audio") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $appDir "Transkript") | Out-Null

$internalDir = Join-Path $appDir "_internal"
$modelsDir = Join-Path $internalDir "models"
$tinyModelOut = Join-Path $modelsDir "faster-whisper-tiny"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

Write-Host "Bundling tiny Whisper model..."
$tinyModelSource = py -3.13 -B -c "from huggingface_hub import snapshot_download; print(snapshot_download('Systran/faster-whisper-tiny'))"
$tinyModelSource = ($tinyModelSource | Select-Object -Last 1).Trim()
if (Test-Path $tinyModelOut) {
    Remove-Item $tinyModelOut -Recurse -Force
}
Copy-Item $tinyModelSource $tinyModelOut -Recurse -Force

Write-Host ""
Write-Host "Build complete:"
Write-Host "  $appDir"
Write-Host ""
Write-Host "Run:"
Write-Host "  $appDir\FraudCallScanner.exe"
