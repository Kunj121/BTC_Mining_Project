import load_data
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import plotly.graph_objects as go
from scipy.interpolate import griddata
import subprocess
import sys

try:
    from arch import arch_model
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "arch"])
    from arch import arch_model


#align columns

#first try comparing da and btc

def align_columns_btc(node = 'NSP'):

    dict = {'NSP':load_data.NSP_NW_da_data(), 'ODEL': load_data.ODELL_da_data()}

    if node == 'NSP':
        node = dict['NSP']
    else:
        node = dict['ODEL']


    btc = load_data.btc_data(time = 'hourly')[['price_close']]
    da = node

    btc.rename(columns = {'price_close':'btc'}, inplace = True)


    rt = load_data.NSP_NW_rt_data()

    dadf = da.copy()
    rtdf = rt.copy()

    rtdf = rt[['interval_end_local', 'lmp']]

    dadf = da[['interval_end_local', 'lmp']]

    rtdf.rename(columns={'interval_end_local':'date', 'lmp':'rt'}, inplace = True)
    dadf.rename(columns={'interval_end_local': 'date', 'lmp':'da'}, inplace=True)

    dadf.set_index('date', inplace=True)
    rtdf.set_index('date', inplace=True)



    merged = pd.merge(dadf, btc, left_index=True, right_index=True)
    merged.index = pd.DatetimeIndex(merged.index, tz='UTC')

    return merged



def simulate_energy(seed = None, plot = True, intraday = False, Node = 'NSP'):

    dabtc = align_columns_btc(node = Node).copy()
    df = dabtc.copy()



    # === Step 1: Historical DA returns ===
    returns = df['da'].pct_change().dropna()
    prices = df['da'].dropna()
    initial_price = prices.iloc[-1]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    # === Step 2: Fit GARCH(1,1) to returns ===
    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero')
    garch_fit = model.fit(disp='off')

    # === Step 3: Set up hourly index and controlled seasonality ===

    if intraday:
        n_paths = 500
        n_hours = 365*24*4
        start_date = pd.Timestamp("2025-07-01 00:00:00")
        hourly_index = pd.date_range(start=start_date, periods=n_hours, freq='15T')
        rng = np.random.default_rng(seed = seed)
        sin_hour  = np.sin(2 * np.pi * ((hourly_index.hour * 60 + hourly_index.minute) / (24*60)))  # 15-min resolution


    else:
        n_paths = 500
        n_hours = 365 * 24
        start_date = pd.Timestamp("2025-07-01 00:00:00")
        hourly_index = pd.date_range(start=start_date, periods=n_hours, freq='H')
        rng = np.random.default_rng(seed = seed)
        # convert to 1‑D ndarray before using in math
        sin_hour  = np.sin(2 * np.pi * hourly_index.hour.to_numpy() / 24*4)  # shape (N,)


    # ---------- per‑path seasonal mean μ_t_paths --------------------------
    base_mu   = dabtc['da'].mean()                       # long‑run level



    amp_paths = rng.integers(3, 11, size=(n_paths, 1))   # each path gets its own amplitude
    sin_hour = np.asarray(sin_hour).flatten()  # Ensure 1D
    mu_t_paths = base_mu + amp_paths * sin_hour[np.newaxis, :]

    # weekend discount
    weekend_mask = hourly_index.weekday.to_numpy() >= 5
    mu_t_paths[:, weekend_mask] *= 0.8

    # optional small per‑path noise
    mu_t_paths += rng.normal(0, 2, size=mu_t_paths.shape)


    np.random.seed(seed)                          # reproducible loop behaviour

    sim_volatility = np.empty((n_paths, n_hours))
    garch_ret      = np.empty((n_paths, n_hours))   # (% → will divide by 100 later)

    for i in range(n_paths):
        sim = model.simulate(garch_fit.params, nobs=n_hours)
        sim_volatility[i, :] = sim['volatility']
        garch_ret[i, :]      = sim['data']          # still in % units

    garch_ret /= 100.0

    # === Step 5: Simulate price paths ===
    phi = 0.5  # very weak mean reversion
    price_paths = np.zeros((n_paths, n_hours))
    price_paths[:, 0] = initial_price

    for t in range(1, n_hours):
        deviation         = price_paths[:, t-1] - mu_t_paths[:, t]
        drift_component   = mu_t_paths[:, t] + phi * deviation
        shock_component   = price_paths[:, t-1] * garch_ret[:, t]
        price_paths[:, t] = drift_component + shock_component

   # === Step 6: Optional shock injection ===
    shock_option = rng.integers(0, 2)
    shock_mask = np.zeros_like(price_paths, dtype=bool)
    shock_decay_hours = 10

    if shock_option == 0:
        for i in range(n_paths):
            num_shocks = rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            shock_hours = rng.choice(np.arange(1, n_hours - (shock_decay_hours + 1)), size=num_shocks, replace=False)

            for t in shock_hours:
                direction = rng.choice([-1, 1])
                base_price = price_paths[i, t-1]

                # Build-up
                price_paths[i, t] = base_price + direction * rng.integers(10, 30)
                shock_mask[i, t] = True

                price_paths[i, t + 1] = price_paths[i, t] + direction * rng.integers(30, 90)
                shock_mask[i, t + 1] = True

                # Decay
                for d in range(1, shock_decay_hours + 1):
                    decay_val = int(100 * (0.5 ** d))
                    adjust = -direction * decay_val
                    price_paths[i, t + 1 + d] = price_paths[i, t + d] + adjust
                    shock_mask[i, t + 1 + d] = True


    if plot:

        # === Step 7 (Alternative): Plot Monthly Averages ===
        plt.figure(figsize=(14, 6))

        for i in range(5):
            # Create a Series with datetime index
            ts = pd.Series(price_paths[i], index=hourly_index)

            # Resample to monthly average
            monthly_avg = ts.resample('H').mean()

            # Plot
            plt.plot(monthly_avg.index, monthly_avg.values, label=f'Path {i+1}', alpha=0.8)

        plt.title('Monthly Average of Simulated DA Prices (Hourly GARCH)')
        plt.xlabel('Month')
        plt.ylabel('Average DA Price ($/MWh)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"Simulated {n_hours} steps of DA")
    return {
        "price_paths": price_paths,           # shape: (n_paths, n_hours)
        "hourly_index": hourly_index,         # pd.DatetimeIndex for each hour
        "mu_t_paths": mu_t_paths,             # deterministic seasonal mean for each path
        # "shock_mask": shock_mask              # boolean mask of where shocks occurred
    }


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_btc_paths(M=500, seed=None, plot=True, intraday=False, Node = 'NSP'):
    """
    Simulate M BTC price paths for N hours (or 15-min intervals if intraday) using bootstrapped historical returns.
    """
    rng = np.random.default_rng(seed)
    df = align_columns_btc(node= Node).copy()

    if intraday:
        freq = '15T'
        N = 365 * 24 * 4
    else:
        freq = 'H'
        N = 365 * 24

    df = df.resample(freq).mean()
    #
    # if intraday:
    #     full_index = pd.date_range(start='2025-07-01', periods=N, freq='15T')
    #     df = df.reindex(full_index).interpolate("linear").ffill().bfill()
    # else:
    #     pass


    btc_returns = df['btc'].pct_change().dropna()
    initial_btc = df['btc'].iloc[-1]

    btc_sim_paths = np.zeros((M, N))
    btc_sim_paths[:, 0] = initial_btc

    for t in range(1, N):
        btc_sim_paths[:, t] = btc_sim_paths[:, t-1] * (1 + rng.choice(btc_returns, size=M, replace=True))

    start_date = pd.Timestamp("2025-07-01")
    date_index = pd.date_range(start=start_date, periods=N, freq=freq)

    # Optional plot
    if plot:
        plt.figure(figsize=(12, 5))
        for i in range(min(10, M)):
            plt.plot(date_index, btc_sim_paths[i], label=f'Path {i+1}', alpha=0.7)

        plt.title('Simulated BTC Price Paths (Bootstrapped Hourly Returns)')
        plt.xlabel('Date')
        plt.ylabel('BTC Price ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"Simulated {N} steps of BTC")

    return btc_sim_paths


# -----------------------------------------------------------
# Basis function for the continuation‑value regression
# -----------------------------------------------------------
def basis(btc_price, power_price, t_idx, HOURS):
    """
    Return a 2‑D array of explanatory variables evaluated at hour t.
    Inputs are *1‑D NumPy arrays* (or pandas Series) of the same length.

    Columns used here:
      1. constant (1)
      2. BTC price
      3. Power price
      4. Interaction term BTC × Power
      5. Time remaining in hours (HOURS - t)

    Feel free to add higher‑order or log terms if they improve the fit.
    """
    return np.column_stack([
        np.ones(len(btc_price)),
        btc_price,
        power_price,
        btc_price * power_price,
        (HOURS - t_idx) * np.ones(len(btc_price))
    ])


def run_sim(seed = None, intraday = False, analysis = False, plot_vol_surface = False, Node = 'NSP'):
    M = 500

    dabtc = align_columns_btc(node= Node)
    dabtc.index = pd.DatetimeIndex(dabtc.index, tz='UTC')




    if intraday:
        btc_paths = simulate_btc_paths(M=M, seed=None, plot=False, intraday=True, Node=Node)

        da_paths = simulate_energy(seed = None, plot = False, intraday = False, Node = 'NSP')['price_paths']



        # Shift DA paths forward by 1 day (24*4 for 15-min intraday)
        lag = 96
        da_paths = np.roll(da_paths, -lag, axis=1)
        da_paths[:, -lag:] = np.nan




    else:
        btc_paths = simulate_btc_paths(
            M=M, seed=None, plot=False, intraday=False, Node=Node
        )

        da_paths = simulate_energy(seed=None, plot=False, intraday=False
        )["price_paths"]


        lag = 24
        da_paths = np.roll(da_paths, -lag, axis=1)
        da_paths[:, -lag:] = np.nan



    valid_horizon = da_paths.shape[1] - lag
    da_paths = da_paths[:, :valid_horizon]
    btc_paths = btc_paths[:, :valid_horizon]


    # Set HOURS and hours based on btc_paths shape
    HOURS = btc_paths.shape[1]
    hours = HOURS / 365  # recovers average hours per day (24 or 24*15)



    # ------------------------------------------------------------------
    # 1) pre‑compute the constant that maps BTC‑USD → breakeven $/MWh
    # ------------------------------------------------------------------
    watts   = 3_250                    # rig draw (W)
    MWh_day = watts * hours / 1_000_000    # 0.00325 MW × 24 h = 0.078 MWh per day

    btc_gen_day = 0.00008              # BTC produced per rig per day
    K = btc_gen_day / MWh_day          # ≈ 0.00008 / 0.078 = 0.00102564  (BTC/MWh)

    # ------------------------------------------------------------------
    # 2) apply to the whole path matrix in one line
    # ------------------------------------------------------------------
    breakeven_paths = btc_paths * K     # shape: (M, H)


    # btc_per_hour_fleet = 0.00008 / hours * 1_000         # 0.00333  BTC/h
    # fleet_MWh_hour     = 3_250 / 1_000 * 1_000 / 1_000  # 3.25 MWh/h

    intr_payoff = (
          breakeven_paths       # $ revenue
        - da_paths             # $ power cost
    )


    disc          = np.exp(-0.04 / 8_760)
    cashflows     = np.zeros_like(intr_payoff)    # (M, H)
    exercise      = np.zeros_like(intr_payoff, dtype=bool)
    exercised_flag= np.zeros(M, dtype=bool)


    for t in range(HOURS - 2, -1, -1):           # start from H‑2

        # 1) ---------------------------------------------------------------
        # universal discount‑carry so nobody is left at zero
        cashflows[:, t] = cashflows[:, t + 1] * disc

        # 2) ---------------------------------------------------------------
        itm = (intr_payoff[:, t] > 0) & (~exercised_flag)
        if not np.any(itm):
            continue

        X = basis(btc_paths[itm, t], da_paths[itm, t], t, HOURS)
        Y = cashflows[itm, t + 1]                 # already discounted
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        cont = X @ coef

        start_now = intr_payoff[itm, t] >= cont
        idx_ex    = np.where(itm)[0][start_now]

        # 3) ---------------------------------------------------------------
        cashflows[idx_ex, t:]  = intr_payoff[idx_ex, t:]
        exercise [idx_ex, t:]  = True
        exercised_flag[idx_ex] = True


    hour_disc  = disc ** np.arange(HOURS)
    npv_lsm    = (cashflows * hour_disc).sum(axis=1).mean()
    std_lsm    = (cashflows * hour_disc).sum(axis=1).std()
    npv_naive  = ((intr_payoff > 0) * intr_payoff * hour_disc).sum(axis=1).mean()
    npv_lsm_paths = cashflows
    #
    print(f"LSMC optimal NPV : ${npv_lsm:,.0f}")
    # print(f"Greedy mine‑when‑positive NPV: ${npv_naive:,.0f}")
    # print(f"Relative uplift : {100*(npv_lsm/npv_naive-1):.1f}%")
    # print(npv_lsm.round(), npv_naive.mean().round())


    if plot_vol_surface:
        window = 24
        btc_vols = []
        da_vols = []
        net_payouts = []

        for idx in np.random.choice(M, size=M, replace=False):

            # Pick random path sample
            btc_series = pd.Series(btc_paths[idx])
            da_series = pd.Series(da_paths[idx])

            # Compute rolling volatilities
            btc_ret = np.log(btc_series / btc_series.shift(1)).dropna()
            da_ret = da_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

            btc_vol = btc_ret.rolling(window).std()
            da_vol = da_ret.rolling(window).std()

            btc_vols.append(btc_vol.mean())
            da_vols.append(da_vol.mean())
            net_payouts.append((cashflows[idx] * hour_disc).sum())

        # Prepare inputs
        X = np.array(btc_vols)
        Y = np.array(da_vols)
        Z = np.array(net_payouts)

        # Create interpolation grid
        xi = np.linspace(X.min(), X.max(), 50)
        yi = np.linspace(Y.min(), Y.max(), 50)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate Z values on the grid
        zi = griddata((X, Y), Z, (xi, yi), method='linear')

        # Create interactive 3D surface plot
        fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Viridis')])

        fig.update_layout(
            title="Net Payout vs BTC & Electricity Volatility (LSMC Surface)",
            scene=dict(
                xaxis_title='BTC Volatility (σ_b)',
                yaxis_title='Electricity Volatility (σ_e)',
                zaxis_title='Net Payout ($)'
            ),
            width=800,
            height=600,
            margin=dict(l=20, r=20, b=40, t=40)
        )

        fig.show()

    if analysis:
        curtail = npv_lsm - npv_naive
        x = pd.Series({
        "Operate all ($)"  : f"{npv_naive:.0f}",
        "Curtail value ($)": f"{curtail:.0f}",
        "Net payout ($)"   : f"{npv_lsm:.0f}",
        })
        intra = pd.DataFrame(x)
        intra = intra.T
        return intra
    else:
        return {
            "npv_lsm"   : npv_lsm,
            "btc_paths" : btc_paths,
            "da_paths"  : da_paths,
            "intrinsic" : intr_payoff,
            "npv_lst_std": std_lsm,
            "cashflows" :(cashflows * hour_disc).sum(axis=1),
    }
















