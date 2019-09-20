import indicators as i
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def series_to_supervised(data, n_in=1,n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
       cols.append(df.shift(i))
       names += ['var{0}(t-{1})'.format(col,i) for col in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
       cols.append(df.shift(-i))
       if i == 0:
         names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
       else:
         names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
      agg.dropna(inplace=True)
    return agg

train_dataset=pd.read_csv('./historical_daily_data/tata_historical.csv')
dataset=i.add_default_indicators(train_dataset)
dataset=dataset.iloc[60:]
dataset.drop(axis=1,columns=['Date','Open','High','Low','Adj Close','Volume'],inplace=True)
print(dataset.shape)
values=dataset.values
values = values.astype('float64')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

n_timesteps=60
n_features=dataset.shape[1]
reframed=series_to_supervised(scaled,n_timesteps)
n_obs=n_timesteps*n_features
print(reframed.shape)
#print(reframed.columns)
train_values=reframed.values
train_X,train_y=train_values[:,:n_obs],train_values[:,-n_features]
print(train_X.shape, len(train_X), train_y.shape)
#print(train_y)
train_X = train_X.reshape((train_X.shape[0], n_timesteps, n_features))

n_cells=100
# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = n_cells, return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = n_cells, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = n_cells, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units =n_cells ,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fifth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units =n_cells ))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(train_X, train_y, epochs = 10, batch_size = 32)

#train_dataset=pd.read_csv('./historical_daily_data/tata_historical.csv')
test_dataset=pd.read_csv('./historical_daily_data/tata_recent.csv')
n_test_obs=test_dataset.shape[0]
dataset_total=train_dataset.append(test_dataset,sort=False) 
test_dataset=dataset_total.iloc[dataset_total.shape[0]-test_dataset.shape[0]-365:]
test_dataset=i.add_default_indicators(test_dataset)
test_dataset=test_dataset.iloc[test_dataset.shape[0]-n_test_obs-n_timesteps:]
test_dataset.drop(axis=1,columns=['Date','Open','High','Low','Adj Close','Volume'],inplace=True)
values=test_dataset.values
values = values.astype('float64')
#scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.transform(values)

#n_timesteps=60
#n_features=test_dataset.shape[1]
reframed=series_to_supervised(scaled,n_timesteps)
#n_obs=n_timesteps*n_features
print(reframed.shape)
print(reframed.columns)
test_values=reframed.values
test_X,test_y=test_values[:,:n_obs],test_values[:,-n_features]
print(test_X.shape, len(test_X), test_y.shape)
#print(test_y)
test_X = test_X.reshape((test_X.shape[0], n_timesteps, n_features))
print(test_X.shape)
#print(test_X[0,1,:])

yhat = regressor.predict(test_X)
test_X=test_X.reshape((test_X.shape[0], n_timesteps*n_features))
yhat=np.concatenate((yhat,test_X[:,-(n_features-1):]),axis=1)
inv_yhat = scaler.inverse_transform(yhat)
inv_yhat=inv_yhat[:,0]
print(yhat.shape,test_X.shape)
print(test_y.shape,test_X.shape)
test_y=test_y.reshape(-1,1)
y=np.concatenate((test_y,test_X[:,-(n_features-1):]),axis=1)
inv_y=scaler.inverse_transform(y)
inv_y=inv_y[:,0]

#print(inv_y)

real_stock_price=inv_y.flatten()
predicted_stock_price=inv_yhat.flatten()
#print(real_stock_price)
#print(predicted_stock_price)
final_result=pd.DataFrame({'real':real_stock_price,'predicted':predicted_stock_price})
final_result.to_csv('Result_indicator.csv')


















