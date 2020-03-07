# TODO 1: Get currency data
# import matplotlib.pyplot as plt
# from alpha_vantage.foreignexchange import ForeignExchange
# cc = ForeignExchange ( key='YOUR_API_KEY', output_format='pandas' )
# #data, meta_data = cc.get_currency_exchange_intraday ( from_symbol='GBP', to_symbol='CNY' )
# data, meta_data = cc.get_currency_exchange_daily(from_symbol = 'GBP', to_symbol= 'CNY', outputsize= 'full')
# print ( data )
# data[ '4. close' ].plot ()
# plt.tight_layout ()
# plt.title ( 'Intraday GBP/CNY' )
# plt.show()
# #


"""
# TODO Step 1: Loading Data
"""
import sys
import math
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time

from keras import Sequential
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout

# TODO 1: Fetching Data


time_start = time.time()

key = 'DMTSPM1P2ZOV4XN5'
stock_id = 'JD'
outputsize = 'full'  # 'full' or 'compact'
datafile = stock_id

model: Sequential = Sequential()


def DEBUG(level, message, type):
    if type == 'message':
        head = 'MSG:'
    else:
        head = 'ERROR:'
    print(head, '*' * level, message)

def loaddata():
    try:
        data = pd.read_csv(datafile, index_col="Date")
        print("Reusing the existing file.")
        print(data.describe())

    except FileNotFoundError:
        print("File is not found. Recreating...")
        ts = TimeSeries(key='DMTSPM1P2ZOV4XN5', output_format='pandas')
        d_, meta_data = ts.get_daily_adjusted(symbol=stock_id, outputsize="full")
        df = d_[::-1]  # Reverse the order inside
        data = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
        for index, row in df.iterrows():
            date = datetime.strptime(str(index), '%Y-%m-%d %H:%M:%S')
            data_row = [date.date(), float(row['3. low']), float(row['2. high']),
                        float(row['4. close']), float(row['1. open'])]
            data.loc[-1, :] = data_row
            data.index = data.index + 1
        data = data.set_index('Date')
        data.to_csv(datafile, index=True)
        data = pd.read_csv(datafile, index_col="Date")
        print("Reusing the existing file.")
        print(data.describe())

    except ValueError or PermissionError:
        print("Application Error, exit.")
        sys.exit(0)

    # Plotting
    plt.tight_layout()
    data.plot()
    plt.axis('auto')
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.title(stock_id)
    plt.show()
    time_end = time.time()
    print("Time cost:", time_end - time_start, 's')

    return data


def preprocess(data):
    """

    :type data: pd.Dataframe
    """
    import statistics

    training_data_len = math.ceil(len(data) * 0.8)
    print('-' * 20)
    print('Train data length...', training_data_len)
    scaler = StandardScaler()
    print('-' * 20)
    print('All data desc...', data.describe())
    print(data.head())
    scaler.fit(data)
    standarlized_X = scaler.transform(data)
    print('-' * 20)
    print('All data...', standarlized_X.shape)
    train_data = standarlized_X[0: training_data_len, :]
    print('-' * 20)
    print('Train Dataset...', train_data.shape)
    test_data = standarlized_X[training_data_len:, :]
    print('-' * 20)
    print('Test Dataset...', test_data.shape)

    x_train = []
    y_train = []
    timestamp = 60
    print('-' * 20)
    for i in range(timestamp, len(train_data)):
        x_train.append((train_data[i - timestamp:i, 0]))
        y_train.append((train_data[i, 0]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print('-' * 20)
    print('x shape ...', x_train.shape)
    print('y shape ...', y_train.shape)
    print('-' * 20)
    print('x_train', x_train)
    print('y', y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print('-' * 20)
    print('x reshape ...', x_train.shape)
    print('y shape ...', y_train.shape)

    x_test = []
    y_test = []
    timestamp = 60
    print('-' * 20)
    for i in range(timestamp, len(test_data)):
        x_test.append((test_data[i - timestamp:i, 0]))
        y_test.append((test_data[i, 0]))

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print('-' * 20)
    print('x shape ...', x_test.shape)
    print('y shape ...', y_test.shape)
    print('-' * 20)
    print('x_test', x_test)
    print('y', y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print('-' * 20)
    print('x reshape ...', x_test.shape)
    print('y shape ...', y_test.shape)

    return x_train, y_train, x_test, y_test


def training(x_train, y_train, x_test, y_test):

    DEBUG(1, 'Model Training', 'message')
    # 1. Model Architecture
    #    with Sequential Model (Just init model)
    #    (Has been declared as global variant)
    #    TODO: Var(model) should be local or global? which one will be better?

    # # 2. and Recurrent Neural Network (RNN - LSTM method)
    # model.add(Embedding(20000, 128))
    # model.add(
    #     LSTM(units=92, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # model.add(LSTM(units=92, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    # model.add(LSTM(units=92, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    # model.add(LSTM(units=92, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    # model.add(Dense(units=1))
    #
    # # 3. Model Fine-tuning
    # #    TODO: Fine Tuning options?
    # from keras.optimizers import RMSprop
    # opt = RMSprop(lr=0.0001, decay=1e-6)
    #
    # # 4. Early Stopping - Callback when loss rate cannot be reduced again ( 2 times continuously)
    # from keras.callbacks import EarlyStopping
    # early_stopping_monitor = EarlyStopping(patience=2)

    # # 5. Compiling Model - Recurrent Neural Network
    # #    TODO: What about the Regression model instead off RNN?
    # model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
    # # model.compile(optimizer='opt', metrics=['accuracy'], loss='mean_squared_error')
    #
    # # 6. Model Training (with validation)
    # model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test),
    #           callbacks=[early_stopping_monitor])
    # return model

def valiating(X_test, y_test):
    pass


def predicting(X, y_actual):
    pass
