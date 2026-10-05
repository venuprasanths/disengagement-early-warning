# PowerShell runner for Windows environments
param (
    [string]$Action = "help"
)

$pythonDir = "C:\Users\acer\AppData\Local\Programs\Python\Python311"
$pythonExe = if (Test-Path "$pythonDir\python.exe") { "$pythonDir\python.exe" } else { "python" }

switch ($Action.ToLower()) {
    "setup" {
        Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
        & $pythonExe -m pip install -r requirements.txt
    }
    "data" {
        Write-Host "Generating synthetic cohort data (500 students, 16 weeks)..." -ForegroundColor Cyan
        & $pythonExe -m data.synthetic_data_generator --output data/synthetic_cohort.csv --students 500 --weeks 16 --seed 42
    }
    "test" {
        Write-Host "Running comprehensive test suite (21 tests)..." -ForegroundColor Cyan
        & $pythonExe -m pytest tests/ -v
    }
    "stress" {
        Write-Host "Running sensitivity and stress testing across cohort compositions..." -ForegroundColor Cyan
        & $pythonExe -m src.sensitivity_analysis
    }
    "drift" {
        Write-Host "Running multi-year longitudinal drift analysis..." -ForegroundColor Cyan
        & $pythonExe -m src.drift_monitor
    }
    "backtest" {
        Write-Host "Running temporal holdout backtesting..." -ForegroundColor Cyan
        & $pythonExe -m src.backtest
    }
    "run" {
        Write-Host "Launching Streamlit trade-off dashboard..." -ForegroundColor Cyan
        & $pythonExe -m streamlit run dashboard/app.py
    }
    default {
        Write-Host "Usage: .\run.ps1 [setup | data | test | stress | drift | backtest | run]" -ForegroundColor Yellow
    }
}
