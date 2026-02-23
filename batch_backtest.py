import sys, os, subprocess, re

stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"]

results = []

for stock in stocks:
    print(f"Running backtest for {stock}...")
    try:
        cmd = ["python", "run_backtest.py", "--source", "db", "--symbol", stock, "--limit", "20000"]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        
        # Parse metrics
        pnl_match = re.search(r"Total P&L:\s+\$\s+([-\d.,]+)", output)
        win_rate_match = re.search(r"Win Rate:\s+([\d.]+)%", output)
        trades_match = re.search(r"Total Trades:\s+(\d+)", output)
        sharpe_match = re.search(r"Sharpe Ratio:\s+([-\d.]+)", output)
        pf_match = re.search(r"Profit Factor:\s+([\d.]+)", output)
        
        if pnl_match and win_rate_match and trades_match:
            pnl = pnl_match.group(1)
            win_rate = win_rate_match.group(1)
            trades = trades_match.group(1)
            sharpe = sharpe_match.group(1) if sharpe_match else "N/A"
            pf = pf_match.group(1) if pf_match else "N/A"
            
            results.append({
                "symbol": stock,
                "trades": trades,
                "win_rate": win_rate,
                "pnl": pnl,
                "sharpe": sharpe,
                "pf": pf
            })
            print(f"  {stock}: Trades={trades}, WinRate={win_rate}%, P&L={pnl}")
        else:
            print(f"  {stock}: Could not parse metrics.")
            
    except subprocess.CalledProcessError as e:
        print(f"  {stock}: Error running backtest -> {e}")

print("\n" + "="*80)
print("BATCH BACKTEST RESULTS")
print("="*80)
print(f"{'Symbol':<15} | {'Trades':<8} | {'Win Rate':<10} | {'P&L':<15} | {'Sharpe':<8} | {'Profit Factor':<12}")
print("-" * 80)
for r in results:
    print(f"{r['symbol']:<15} | {r['trades']:<8} | {r['win_rate']:>6}%   | ${r['pnl']:<14} | {r['sharpe']:<8} | {r['pf']:<12}")
print("="*80)

