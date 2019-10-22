import pandas as pd

f_df=pd.read_csv('./historical_daily_data/SBIN.EO.csv')
historical_df=f_df.iloc[0:-365]
recent_df=f_df.iloc[-365:]
historical_df.to_csv('./historical_daily_data/sbin_historical.csv')
recent_df.to_csv('./historical_daily_data/sbin_recent.csv')
