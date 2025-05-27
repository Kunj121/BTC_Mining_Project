# 🧠 BTC Mining Optimization Project

## 🏗️ Overview
This project models and optimizes a Bitcoin (BTC) mining data center's operations using historical BTC prices and MISO electricity market data. The objective is to recommend an operating strategy that maximizes annual profit while accounting for power market volatility, rig efficiency, and financial risk.
---

## 📍 Project Scope
To run the project, please clone the repo, and run BTC_Mining_Simulation.ipynb.


### 🔹 Part 1: Optimize Existing Facility
Your uncle owns a BTC mining facility with:
- **1,000 Antminer S19 Pro rigs**
- Operating in the **MISO NSP.NWELOAD** node
- Exposed to **Day-Ahead (DA)** and **Real-Time (RT)** electricity markets

Tasks:
- Simulate mining profitability using LSMC
- Estimate profit for July 1, 2025 – June 30, 2026
- Analyze curtailment, breakeven conditions, and volatility surfaces

### 🔹 Part 2: Expansion Evaluation
Your uncle considers building a second facility. You will:
- Choose a new MISO node or expand the original
- Compare profitability and volatility of both investments

---

## 📊 Assumptions

- **Risk-free rate**: 4% annualized  
- **Mining rate per Antminer**: 0.00008 BTC/day  
- **Power usage per Antminer**: 3,250W  
- **Breakeven electricity cost**: $95/MWh at $93,000 BTC/USD  
- **BTC mining difficulty**: Constant  
- **Revenue = BTC; Cost = electricity only**  
- **No ancillary service revenue**  

---

## 📈 Key Outputs

- ✅ Expected annual profit across 500+ simulations  
- 📉 Volatility of profitability  
- 📊 Profit-to-volatility ratio  
- 🌐 3D surface plots: BTC volatility (σ_b), electricity volatility (σ_e), and net payout  
- ⏱️ Intraday profitability analysis at hourly, 15-min, and 5-min intervals

---
