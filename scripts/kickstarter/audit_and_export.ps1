# Audit, validate, and export Kickstarter SQLite for analysis (Windows/local).
# Usage: .\scripts\kickstarter\audit_and_export.ps1 [-DbPath PATH] [-InputCsv PATH]

param(
    [string]$DbPath = "data/kickstarter/kickstarter_main.db",
    [string]$InputCsv = "data/my_file.csv"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

Write-Host "--- Kickstarter audit + export ---"
Write-Host "DB: $DbPath"

uv run python src/processing/audit_completeness.py --db $DbPath
uv run python src/processing/validate_kickstarter_for_analysis.py --db $DbPath --input-csv $InputCsv --strict
uv run python src/processing/export_sqlite_for_analysis.py --db $DbPath

Write-Host "--- Done ---"
