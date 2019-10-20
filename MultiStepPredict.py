import LSTM as lstm
import pandas as pd
from copy import deepcopy
import indicators as i

next_index=0
n_timesteps=180
n_steps=15

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
    close_scaler=lstm.load_scaler('tm_close_minmax.pkl')
    close_lstm=lstm.load_lstm('tm_close_lstm.h5')

    open_scaler=lstm.load_scaler('tm_open_minmax.pkl')
    open_lstm=lstm.load_lstm('tm_open_lstm.h5')

    high_scaler=lstm.load_scaler('tm_high_minmax.pkl')
    high_lstm=lstm.load_lstm('tm_high_lstm.h5')

    low_scaler=lstm.load_scaler('tm_low_minmax.pkl')
    low_lstm=lstm.load_lstm('tm_low_lstm.h5')

    historical_dataset=pd.read_csv('./historical_daily_data/tata_historical.csv')
    historical_dataset=historical_dataset.tail(365)
    recent_dataset=pd.read_csv('./historical_daily_data/tata_recent.csv')
    #next_index=recent_dataset.shape[0]-2



    c_df=pd.DataFrame()
    while True:
        row=get_next_data(recent_dataset)
        print(next_index)
        print(row)

        if not row:
            break
        historical_dataset=historical_dataset.append(row,ignore_index=True)
        working_dataset=deepcopy(historical_dataset)
        close_prices=[]

        for idx in range(n_steps):
            close_dataset, n_features = lstm.prepare_data(input_df=working_dataset, y='Close', run_function=i.add_default_indicators,
                                                          drop_columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'],
                                                          start_row=180)
            predicted_close=lstm.predict(close_lstm,close_dataset,n_features,n_timesteps,close_scaler)

            open_dataset, n_features = lstm.prepare_data(input_df=working_dataset, y='Open', run_function=i.add_default_indicators,
                                                         drop_columns=['Date', 'Close', 'High', 'Low', 'Adj Close', 'Volume'],
                                                         start_row=180)
            predicted_open = lstm.predict(open_lstm, open_dataset, n_features, n_timesteps, open_scaler)

            high_dataset, n_features = lstm.prepare_data(input_df=working_dataset, y='High', run_function=i.add_default_indicators,
                                                         drop_columns=['Date', 'Close', 'Open', 'Low', 'Adj Close', 'Volume'],
                                                         start_row=180)
            predicted_high = lstm.predict(high_lstm, high_dataset, n_features, n_timesteps, high_scaler)

            low_dataset, n_features = lstm.prepare_data(input_df=working_dataset, y='Low',
                                                         run_function=i.add_default_indicators,
                                                         drop_columns=['Date', 'Close', 'Open', 'High', 'Adj Close', 'Volume'],
                                                         start_row=180)
            predicted_low = lstm.predict(low_lstm, low_dataset, n_features, n_timesteps, low_scaler)

            o,h,l,c=predicted_open[-1],predicted_high[-1],predicted_low[-1],predicted_close[-1]

            h=max(o,h,l,c)
            l=min(o,h,l,c)
            v=0
            adj_c=0
            pd='T{}'.format(idx+6)
            row.update({pd:c})
            close_prices.append(c)
            working_dataset=working_dataset.append({'Open':o,'Close':c,'High':h,'Low':l,'Adj Close':adj_c,'Volume':v},ignore_index=True)
        print(row)
        c_df=c_df.append(row,ignore_index=True)
   
    c_df.to_csv('tata_motors_prediction_ns.csv')
    for idx in range(n_steps):
        pd='T{}'.format(idx+6)
        c_df[pd]=c_df[pd].shift(idx+6)
        
    c_df.to_csv('tata_motors_prediction_s.csv')








