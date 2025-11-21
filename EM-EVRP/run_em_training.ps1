<#
run_em_training.ps1
PowerShell helper script to launch a training run for EM-EVRP.

Usage examples (PowerShell):
  # Run with defaults
  .\run_em_training.ps1

  # Override defaults
  .\run_em_training.ps1 -Nodes 10 -ChargingNum 4 -TrainSize 1280000 -Iterations 100 -BatchSize 1024

Notes:
- This script assumes you already activated the conda/venv environment that
  contains the project's Python dependencies.
- Adjust `TrainSize`, `Iterations` and `BatchSize` to match your available
  compute resources. For GPU, larger batch sizes (512-2048) may be OK;
  for CPU choose much smaller (16-128).
#>

param(
    [int]$Nodes = 10,
    [int]$ChargingNum = 4,
    [int]$TrainSize = 100000,
    [int]$ValidSize = 5000,
    [int]$Iterations = 100,
    [int]$BatchSize = 256,
    [int]$Seed = 12345,
    [string]$Model = "DRL",
    [string]$Baseline = "rollout",
    [int]$PlotNum = 0
)

Write-Host "Starting EM-EVRP training"
Write-Host " Nodes: $Nodes | Charging stations: $ChargingNum | Train-size: $TrainSize | Iterations: $Iterations | Batch: $BatchSize"

# Move to script directory to run run.py from EM-EVRP
Set-Location -Path $PSScriptRoot

# Build command line
$cmd = @(
    'python', '.\run.py',
    '--nodes', $Nodes,
    '--charging_num', $ChargingNum,
    '--train-size', $TrainSize,
    '--valid-size', $ValidSize,
    '--iterations', $Iterations,
    '--batch_size', $BatchSize,
    '--seed', $Seed,
    '--model', $Model,
    '--baselines', $Baseline,
    '--plot_num', $PlotNum
) -join ' '

Write-Host "Running:" $cmd

# Execute the command
Invoke-Expression $cmd

Write-Host "Training process finished (or exited). Check ExperimentalLog and ExperimentalData for outputs."