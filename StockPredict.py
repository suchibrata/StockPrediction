import pandas as pd
import matplotlib.pyplot as plt
tata_motors=pd.read_csv('./historical_daily_data/TATAMOTORS.BO.E.csv')
#print(tata_motors['Open'])
#tata_motors['Open'].plot()
#plt.show()

tata_motors_recent=tata_motors.tail(365)
tata_motors_recent.to_csv('./historical_daily_data/tata_recent.csv',index=False)
tata_motors_historical=tata_motors.iloc[:-365]
tata_motors_historical.to_csv('./historical_daily_data/tata_historical.csv',index=False)

tata_motors_recent['Open'].plot()
plt.show()