# sweep.ps1
$ErrorActionPreference = "Stop"

# Limit parallelism to avoid hammering the API
$MaxParallel = 3

# Commands to run
$cmds = @(
  @{ name="A_base"; args="--market_type 5m --days 1 --limit-rounds 400 --include-orderbook --step-seconds 20 --max-workers 2 --tau-min 60 --tau-max 240 --base-size 5 --fee-bps 20 --slippage-bps 10 --spread-bps 0 --gas-usd 0 --min-ev 0.002 --max-entry-price 0.65 --min-payout 0.35 --max-spread 0.03 --execution-mode orderbook --max-position-usdc 10 --max-loss-per-day 25 --max-loss-streak-k 5 --max-loss-streak-cooldown 5 --edge-ref 0.05 --min-size 0 --min-edge 0 --min-z 0 --dump-trades trades_fast_base.csv --report-json report_fast_base.json" }
  @{ name="B_spread002"; args="--market_type 5m --days 1 --limit-rounds 400 --include-orderbook --step-seconds 20 --max-workers 2 --tau-min 60 --tau-max 240 --base-size 5 --fee-bps 20 --slippage-bps 10 --spread-bps 0 --gas-usd 0 --min-ev 0.002 --max-entry-price 0.65 --min-payout 0.35 --max-spread 0.02 --execution-mode orderbook --max-position-usdc 10 --max-loss-per-day 25 --max-loss-streak-k 5 --max-loss-streak-cooldown 5 --edge-ref 0.05 --min-size 0 --min-edge 0 --min-z 0 --dump-trades trades_fast_spread002.csv --report-json report_fast_spread002.json" }
  @{ name="C_spread0015"; args="--market_type 5m --days 1 --limit-rounds 400 --include-orderbook --step-seconds 20 --max-workers 2 --tau-min 60 --tau-max 240 --base-size 5 --fee-bps 20 --slippage-bps 10 --spread-bps 0 --gas-usd 0 --min-ev 0.002 --max-entry-price 0.65 --min-payout 0.35 --max-spread 0.015 --execution-mode orderbook --max-position-usdc 10 --max-loss-per-day 25 --max-loss-streak-k 5 --max-loss-streak-cooldown 5 --edge-ref 0.05 --min-size 0 --min-edge 0 --min-z 0 --dump-trades trades_fast_spread0015.csv --report-json report_fast_spread0015.json" }
  @{ name="D_ev0004"; args="--market_type 5m --days 1 --limit-rounds 400 --include-orderbook --step-seconds 20 --max-workers 2 --tau-min 60 --tau-max 240 --base-size 5 --fee-bps 20 --slippage-bps 10 --spread-bps 0 --gas-usd 0 --min-ev 0.004 --max-entry-price 0.65 --min-payout 0.35 --max-spread 0.02 --execution-mode orderbook --max-position-usdc 10 --max-loss-per-day 25 --max-loss-streak-k 5 --max-loss-streak-cooldown 5 --edge-ref 0.05 --min-size 0 --min-edge 0 --min-z 0 --dump-trades trades_fast_ev0004.csv --report-json report_fast_ev0004.json" }
  @{ name="E_tau120_180"; args="--market_type 5m --days 1 --limit-rounds 400 --include-orderbook --step-seconds 20 --max-workers 2 --tau-min 120 --tau-max 180 --base-size 5 --fee-bps 20 --slippage-bps 10 --spread-bps 0 --gas-usd 0 --min-ev 0.002 --max-entry-price 0.65 --min-payout 0.35 --max-spread 0.02 --execution-mode orderbook --max-position-usdc 10 --max-loss-per-day 25 --max-loss-streak-k 5 --max-loss-streak-cooldown 5 --edge-ref 0.05 --min-size 0 --min-edge 0 --min-z 0 --dump-trades trades_fast_tau120_180.csv --report-json report_fast_tau120_180.json" }
)

# Helper to start a job for one command
function Start-RunJob($name, $args) {
  $log = "logs\$name.log"
  New-Item -ItemType Directory -Force -Path "logs" | Out-Null
  Start-Job -Name $name -ScriptBlock {
    param($a, $l)
    # Run and capture stdout+stderr into a log file
    & python -m backtest.run_backtest $a 2>&1 | Tee-Object -FilePath $l
  } -ArgumentList $args, $log
}

# Run with limited parallelism
$jobs = @()
foreach ($c in $cmds) {
  while ((Get-Job -State Running).Count -ge $MaxParallel) {
    Start-Sleep -Seconds 2
  }
  $jobs += Start-RunJob $c.name $c.args
  Write-Host "Started $($c.name)"
}

Write-Host "Waiting for all jobs..."
Wait-Job $jobs | Out-Null
Write-Host "Done. Logs in .\logs\"
Get-Job | Receive-Job | Out-Null
