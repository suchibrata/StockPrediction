import LSTM as lstm
import pandas as pd
import indicators as i

n_timesteps=60
dataset=pd.read_csv('./historical_daily_data/tata_historical.csv')

close_dataset,n_features=lstm.prepare_data(input_df=dataset,y='Close',run_function=i.add_default_indicators,
                                drop_columns=['Date','Open','High','Low','Adj Close','Volume'],
                                start_row=60)
scaled_data=lstm.create_scaled_data(input_df=close_dataset,model_file='tm_close_minmax.pkl')
reframed=lstm.series_to_supervised(data=scaled_data,n_in=n_timesteps)
lstm_regressor=lstm.build_lstm_model(input_shape=(n_timesteps,n_features))
lstm_regressor=lstm.fit_lstm_model(model_file='tm_close_lstm.h5',regressor=lstm_regressor,supervised_df=reframed,
                                   n_features=n_features,n_timesteps=n_timesteps,epochs=10)


open_dataset,n_features=lstm.prepare_data(input_df=dataset,y='Open',run_function=i.add_default_indicators,
                                drop_columns=['Date','Close','High','Low','Adj Close','Volume'],
                                start_row=60)
scaled_data=lstm.create_scaled_data(input_df=close_dataset,model_file='tm_open_minmax.pkl')
reframed=lstm.series_to_supervised(data=scaled_data,n_in=n_timesteps)
lstm_regressor=lstm.build_lstm_model(input_shape=(n_timesteps,n_features))
lstm_regressor=lstm.fit_lstm_model(model_file='tm_open_lstm.h5',regressor=lstm_regressor,supervised_df=reframed,
                                   n_features=n_features,n_timesteps=n_timesteps,epochs=10)


high_dataset,n_features=lstm.prepare_data(input_df=dataset,y='High',run_function=i.add_default_indicators,
                                drop_columns=['Date','Close','Open','Low','Adj Close','Volume'],
                                start_row=60)
scaled_data=lstm.create_scaled_data(input_df=close_dataset,model_file='tm_high_minmax.pkl')
reframed=lstm.series_to_supervised(data=scaled_data,n_in=n_timesteps)
lstm_regressor=lstm.build_lstm_model(input_shape=(n_timesteps,n_features))
lstm_regressor=lstm.fit_lstm_model(model_file='tm_high_lstm.h5',regressor=lstm_regressor,supervised_df=reframed,
                                   n_features=n_features,n_timesteps=n_timesteps,epochs=10)

low_dataset,n_features=lstm.prepare_data(input_df=dataset,y='Low',run_function=i.add_default_indicators,
                                drop_columns=['Date','Close','Open','High','Adj Close','Volume'],
                                start_row=60)
scaled_data=lstm.create_scaled_data(input_df=close_dataset,model_file='tm_low_minmax.pkl')
reframed=lstm.series_to_supervised(data=scaled_data,n_in=n_timesteps)
lstm_regressor=lstm.build_lstm_model(input_shape=(n_timesteps,n_features))
lstm_regressor=lstm.fit_lstm_model(model_file='tm_low_lstm.h5',regressor=lstm_regressor,supervised_df=reframed,
                                   n_features=n_features,n_timesteps=n_timesteps,epochs=10)

