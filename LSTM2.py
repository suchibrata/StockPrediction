import indicators as i
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from copy import deepcopy

def build_lstm_model(input_shape,n_cells=100,n_layers=5,r_dropout=0.2):
    regressor = Sequential()

    if n_layers > 1:
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=n_cells, return_sequences=True, input_shape=input_shape))
        regressor.add(Dropout(r_dropout))

        for l in range(n_layers-2):
           regressor.add(LSTM(units=n_cells, return_sequences=True))
           regressor.add(Dropout(r_dropout))

        #Adding final LSTM layer and Dropout regularization
        regressor.add(LSTM(units=n_cells))
        regressor.add(Dropout(r_dropout))
    else:
        regressor.add(LSTM(units=n_cells,input_shape=input_shape))
        regressor.add(Dropout(r_dropout))

    # Adding the output layer
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    return regressor

def prepare_data(input_df,y,run_function=None,drop_columns=[],start_row=0,end_row=0):
    input_df1=deepcopy(input_df)
    aug_input_df=run_function(input_df1) if run_function and callable(run_function) else input_df1

    if drop_columns:
        aug_input_df.drop(axis=1, columns=drop_columns, inplace=True)

    y_df=pd.DataFrame(aug_input_df[y])
    aug_input_df.drop(axis=1,columns=[y],inplace=True)
    y_df.join(aug_input_df)

    end_row=y_df.shape[0]-end_row if end_row ==0 else end_row
    aug_input_df=y_df.iloc[start_row:end_row]
    return aug_input_df,aug_input_df.shape[1]

def create_scaled_data(input_df,model_file):
    values = input_df.values
    values = values.astype('float64')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    with open(model_file,'wb') as f:
        pickle.dump(scaler,f)
    return scaled

def series_to_supervised(data, n_in=1,n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
       cols.append(df.shift(i))
       names += ['var{0}(t-{1})'.format(col,i) for col in range(n_vars)]

    # forecast sequence t(n_out-1)
    cols.append(df.shift(-n_out+1))
    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    #put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
      agg.dropna(inplace=True)
    return agg

def fit_lstm_model(model_file,regressor,supervised_df,n_features,n_timesteps,epochs=1,batch_size=32):
    train_values = supervised_df.values
    train_X, train_y = train_values[:, :n_features*n_timesteps], train_values[:, -n_features]
    train_X = train_X.reshape((train_X.shape[0], n_timesteps, n_features))

    regressor.fit(train_X, train_y, epochs=epochs, batch_size=batch_size)
    regressor.save(model_file)

    return regressor



def load_scaler(scaler_file):
    with open(scaler_file,'rb') as f:
        scaler=pickle.load(f)
    return scaler

def load_lstm(lstm_file):
    lstm=load_model(lstm_file)
    return lstm

def predict(lstm_regressor,x,n_features,n_timesteps,scaler):
    x_values=x.values
    x_values = x_values.astype('float64')
    x_scaled=scaler.transform(x_values)
    x_df=series_to_supervised(x_scaled,n_timesteps)
    pred_x_values=x_df.values
    pred_x,pred_y=pred_x_values[:,:n_features*n_timesteps],pred_x_values[:,-n_features]
    pred_x=pred_x.reshape(pred_x.shape[0],n_timesteps,n_features)
    yhat=lstm_regressor.predict(pred_x)
    pred_x = pred_x.reshape((pred_x.shape[0], n_timesteps * n_features))
    yhat = np.concatenate((yhat, pred_x[:, -(n_features - 1):]), axis=1)
    inv_yhat = scaler.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, 0]
    predicted_stock_price = inv_yhat.flatten()
    return predicted_stock_price







