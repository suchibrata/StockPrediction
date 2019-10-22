import LSTM as lstm
import pandas as pd
from copy import deepcopy
import indicators as i

next_index=0
n_timesteps=180
n_steps=15
start_row=180

def get_next_data(recent_dataset):
    global next_index
    print(next_index)
    print(recent_dataset.shape[0])
    if recent_dataset.shape[0] > next_index:
        row=recent_dataset.iloc[next_index]
        row=row.to_dict()
        print(row)
        #row=row[0]
        next_index+=1
    else:
        row=None
    return row


if __name__=='__main__':
    close_scaler=[]
    close_lstm=[]
    for idx in range(n_steps):
        cs=lstm.load_scaler('sbin_close_minmax_t{}.pkl'.format(idx+1))
        cl=lstm.load_lstm('sbin_close_lstm.h5'.format(idx+1))
        close_scaler.append(cs)
        close_lstm.append(cl)

    historical_dataset=pd.read_csv('./historical_daily_data/sbin_historical.csv')
    historical_dataset=historical_dataset.tail(365)
    recent_dataset=pd.read_csv('./historical_daily_data/sbin_recent.csv')
    #next_index=recent_dataset.shape[0]-2

    c_df = pd.DataFrame()
    while True:
        row = get_next_data(recent_dataset)
        #print(next_index)
        #print(row)

        if not row:
            break
        historical_dataset = historical_dataset.append(row, ignore_index=True)
        #working_dataset = deepcopy(historical_dataset)
        #close_prices = []

        for idx in range(n_steps):
            close_dataset, n_features = lstm.prepare_data(input_df=historical_dataset, y='Close', run_function=i.add_default_indicators,
                                                          drop_columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'],
                                                          start_row=start_row)

            predicted_close = lstm.predict(close_lstm[idx], close_dataset, n_features, n_timesteps, close_scaler[idx])
            c=predicted_close[-1]
            pd = 'T{}'.format(idx + 1)
            row.update({pd: c})

        c_df = c_df.append(row, ignore_index=True)

    c_df.to_csv('sbin_prediction_ns.csv')
    for idx in range(n_steps):
        pd = 'T{}'.format(idx + 1)
        c_df[pd] = c_df[pd].shift(idx + 1)

    c_df.to_csv('sbin_prediction_s.csv')


