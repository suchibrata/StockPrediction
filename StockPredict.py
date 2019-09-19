import pandas as pd
import matplotlib.pyplot as plt
import indicators as i
tata_motors=pd.read_csv('./historical_daily_data/TATAMOTORS.BO.E.csv')

#tata_motors=i.add_default_indicators(df=tata_motors)

tata_motors_recent=tata_motors.tail(365)
tata_motors_recent.to_csv('./historical_daily_data/tata_recent.csv',index=False)
tata_motors_historical=tata_motors.iloc[:-365]
tata_motors_historical.to_csv('./historical_daily_data/tata_historical.csv',index=False)
tata_motors_recent['Close'].plot()
plt.show()