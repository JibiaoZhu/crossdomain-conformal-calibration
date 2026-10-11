param(
  [string]$Python = "E:\AutosciOpencodeProcessionDocument\venv-exp\Scripts\python.exe",
  [string]$Config = "config.yaml",
  [int]$Seed = 0,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $here
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& $Python run_ablation.py --config $Config --seed $Seed @Rest
