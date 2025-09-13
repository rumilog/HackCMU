# PowerShell script to start the backend server
# This script handles port conflicts and starts the backend properly

Write-Host "🚀 Starting Lantern Fly Tracker Backend..." -ForegroundColor Green

# Check if port 5000 is in use
$portInUse = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "⚠️  Port 5000 is already in use. Killing existing process..." -ForegroundColor Yellow
    $processId = $portInUse.OwningProcess
    Stop-Process -Id $processId -Force
    Start-Sleep 2
}

# Change to backend directory
Set-Location backend

# Start the development server
Write-Host "📦 Starting backend server..." -ForegroundColor Blue
npm run dev
