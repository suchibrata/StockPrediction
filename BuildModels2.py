import LSTM2 as lstm
import pandas as pd
import indicators as i

n_timesteps=180
dataset=pd.read_csv('./historical_daily_data/sbin_historical.csv')
n_steps=15
start_row=180
epochs=50

for idx in range(n_steps):
    close_dataset,n_features=lstm.prepare_data(input_df=dataset,y='Close',run_function=i.add_default_indicators,
                                    drop_columns=['Date','Open','High','Low','Adj Close','Volume'],
                                    start_row=start_row)
    scaled_data=lstm.create_scaled_data(input_df=close_dataset,model_file='sbin_close_minmax_t{}.pkl'.format(idx+1))
    reframed=lstm.series_to_supervised(data=scaled_data,n_in=n_timesteps,n_out=idx+1)
    lstm_regressor=lstm.build_lstm_model(input_shape=(n_timesteps,n_features))
    lstm_regressor=lstm.fit_lstm_model(model_file='sbin_close_lstm_t{}.h5'.format(idx+1),regressor=lstm_regressor,
                                       supervised_df=reframed,n_features=n_features,n_timesteps=n_timesteps,epochs=epochs)
