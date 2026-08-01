# Promote recompiled DB to canonical kickstarter_main.db (with backup).
# Usage: .\scripts\kickstarter\promote_recompiled_db.ps1 [-RecompiledPath PATH] [-MainPath PATH]

param(
    [string]$RecompiledPath = "data/kickstarter/kickstarter_recompiled.db",
    [string]$MainPath = "data/kickstarter/kickstarter_main.db"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$ArchiveDir = "data/kickstarter/archive_dbs"
if (-not (Test-Path $RecompiledPath)) {
    throw "Recompiled DB not found: $RecompiledPath"
}

New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (Test-Path $MainPath) {
    $backup = Join-Path $ArchiveDir "kickstarter_main_$stamp.db"
    Write-Host "Backing up $MainPath -> $backup"
    Move-Item -Force $MainPath $backup
}

Write-Host "Promoting $RecompiledPath -> $MainPath"
Move-Item -Force $RecompiledPath $MainPath
Write-Host "Done. Run scripts/kickstarter/audit_and_export.ps1 to validate and export."
