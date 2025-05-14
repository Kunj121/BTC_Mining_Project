import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def NSP_NW_da_data():
    da = pd.read_parquet('data/DA_MISO_NSP_NWELOAD.parquet')
    da['interval_start_local'] = pd.to_datetime(da['interval_start_local'])
    da['interval_end_local'] = pd.to_datetime(da['interval_end_local'])
    da['interval_start_utc'] = pd.to_datetime(da['interval_start_utc'])
    da.sort_values('interval_start_local', inplace=True)
    return da


def NSP_NW_rt_data():
    rt = pd.read_parquet('data/RT_MISO_NSP_NWRT.parquet')
    rt['interval_start_local'] = pd.to_datetime(rt['interval_start_local'])
    rt['interval_end_local'] = pd.to_datetime(rt['interval_end_local'])
    rt['interval_start_utc'] = pd.to_datetime(rt['interval_start_utc'])
    rt.sort_values('interval_start_local', inplace=True)

    return rt

import requests
def get_btc_price():
    # url = "https://api.coingecko.com/api/v3/simple/price"
    # params = {
    #     'ids': 'bitcoin',
    #     'vs_currencies': 'usd'
    # }
    # response = requests.get(url, params=params)
    # print(response.json())
    # data = response.json()
    # return data['bitcoin']['usd']
    btc_price = 105_000
    return btc_price

# Example usage
# btc_price = get_btc_price()
# print(f"Current BTC Price: ${btc_price}")


def btc_data(time = 'daily'):
    x = pd.read_parquet('data/btc_data.parquet')
    x = x.reset_index()
    x.rename(columns={'index': 'time_end'}, inplace=True)
    x['time_end'] = pd.to_datetime(x['time_end'])
    x.drop(columns=['time_open', 'time_close'], inplace=True)
    # x['time_open'] = pd.to_datetime(x['time_open'])
    # x['time_close'] = pd.to_datetime(x['time_close'])
    x = x.iloc[1:]
    x.set_index('time_end', inplace=True)
    x = x[['price_close', 'returns', 'normalized_ret', 'volume_traded']]

    if time == 'daily':
        x = x.resample('D').last()
    else:
        pass
    return x
