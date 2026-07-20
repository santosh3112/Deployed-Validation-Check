# RPA Bot Monitoring Dashboard - PowerShell Launcher
# Run this if start_server.bat doesn't work

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  RPA Bot Monitoring Dashboard" -ForegroundColor Cyan  
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Move to script directory
Set-Location $PSScriptRoot

# Kill any stale process on port 5000
$stale = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if ($stale) {
    Write-Host "[INFO] Killing stale process on port 5000..." -ForegroundColor Yellow
    $stale | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Milliseconds 500
}

# Check Flask
try {
    python -c "import flask" 2>$null
    Write-Host "[OK] Flask is installed" -ForegroundColor Green
} catch {
    Write-Host "[INFO] Installing Flask..." -ForegroundColor Yellow
    pip install flask werkzeug
}

Write-Host ""
Write-Host "[STARTING] http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "[INFO] Keep this window open while using the app" -ForegroundColor Yellow
Write-Host "[INFO] Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python app.py

Write-Host ""
Write-Host "[STOPPED] Server has stopped." -ForegroundColor Red
Read-Host "Press Enter to close"
