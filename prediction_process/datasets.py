import pandas as pd
import numpy as np

df = pd.read_fwf('datasets/lorenz.txt')
x_train = np.float32(np.array(df.iloc[:, 0])[3000:13000])
y_true = np.float32(np.array(df.iloc[:, 0])[13000:15500])

dfr = pd.read_csv('datasets/bitcoin.csv', usecols=['Close Time', 'Close'], parse_dates=['Close'])
dfr.rename(columns={"Close Time": 'close_time'}, inplace=True)
dfr.close_time = pd.to_datetime(dfr.close_time, unit='ms')
dfr.close_time = dfr.close_time.dt.round(freq='S')
dfr = dfr.set_index('close_time')
dfr.Close = dfr.Close.astype('float')
dfr = dfr[::10]
bitcoin_train = dfr.iloc[:10000].values.reshape(1, -1)[0]
bitcoin_test = dfr.iloc[10000:].values.reshape(1, -1)[0]

el = pd.read_csv('datasets/PJM_Load_hourly.csv', parse_dates=True)
el.Datetime = pd.to_datetime(el.Datetime)
el.set_index('Datetime', inplace=True)
el = el.values.reshape(1, -1)[0]

el_train = el[:20000]
el_test = el[20000:25000]

# normalize train and test datasets
el_train = ((el_train - el_train.min()) / (el_train.max() - el_train.min()))
el_test = ((el_test - el_test.min()) / (el_test.max() - el_test.min()))
