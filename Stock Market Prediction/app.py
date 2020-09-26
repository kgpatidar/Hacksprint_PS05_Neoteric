from numpy import array
import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from numpy.random import seed
import time
import copy
import chainer
import chainer.functions as F
import chainer.links as L
from plotly import tools
from keras.models import load_model
from plotly import graph_objs as go
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl


image_path = ('image.jpg')
st.image(image_path, use_column_width=True)

st.title("Finock ~ Stock Price Predictor 📈")
st.header("Welcome to Finock!")
st.markdown(
    "In this Deep Learning application, we have used the historical stock price data for HDFC to forecast their price for the next 10 days.")


DATA_URL = ('./DATASETS/'+'HDFC'+'.csv')


def load_data():
    data = pd.read_csv(DATA_URL)
    return data


data = load_data()

new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

train = new_data[:4112]
test = new_data[4112:]

scaler = MinMaxScaler(feature_range=(0, 1))
values = new_data.values

values = np.array(values)
values = scaler.fit_transform(values)


x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(values[i-60:i, 0])
    y_train.append(values[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

inputs = new_data[len(new_data) - len(test) - 60:].values
inputs = inputs.reshape(-1, 1)


X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)


X_test = np.reshape(X_test, (1029, 60, 1))


x_train = np.reshape(x_train, (4052, 60, 1))
X_test = np.reshape(X_test, (1029, 60, 1))


model = load_model('HDFC_Model.h5')

x_input = X_test[340].reshape(1, -1)

x_input = np.asarray(x_input[0])

temp_input = x_input.tolist()


def plot_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.Date, y=data['Open'], name="stock_open", line_color='deepskyblue'))
    fig.add_trace(go.Scatter(
        x=data.Date, y=data['Close'], name="stock_close", line_color='dimgray'))
    fig.layout.update(
        title_text='Opening and Closing Price of Stock', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig


plot_fig()

st.header('Candelstick Analyser')

fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'])])
st.plotly_chart(fig)

lst_output = []
n_steps = 60
i = 0
while(i < 11):
    if(len(temp_input) > 60):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print
        ("{} day output {}".format(i, scaler.inverse_transform(yhat)))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i+1

st.header('Stock Prices Predictor')

option = st.selectbox('Decide the Day of Prediction', [
    'Day-1', 'Day-2', 'Day-3', 'Day-4', 'Day-5', 'Day-6', 'Day-7', 'Day-8', 'Day-9', 'Day-10', ])

if (option == 'Day-1'):
    st.write('Closing price of HDFC will be ' + str(lst_output[0]))
elif (option == 'Day-2'):
    st.write('Closing price of HDFC will be ' + str(lst_output[1]))
elif (option == 'Day-3'):
    st.write('Closing price of HDFC will be ' + str(lst_output[2]))
elif (option == 'Day-4'):
    st.write('Closing price of HDFC will be ' + str(lst_output[3]))
elif (option == 'Day-5'):
    st.write('Closing price of HDFC will be ' + str(lst_output[4]))
elif (option == 'Day-6'):
    st.write('Closing price of HDFC will be ' + str(lst_output[5]))
elif (option == 'Day-7'):
    st.write('Closing price of HDFC will be ' + str(lst_output[6]))
elif (option == 'Day-8'):
    st.write('Closing price of HDFC will be ' + str(lst_output[7]))
elif (option == 'Day-9'):
    st.write('Closing price of HDFC will be ' + str(lst_output[8]))
elif (option == 'Day-10'):
    st.write('Closing price of HDFC will be ' + str(lst_output[9]))


st.header('Prediction Graph for the next 10 Days')

day_new = np.arange(1, 61)
day_pred = np.arange(61, 72)
st.line_chart(day_new, scaler.inverse_transform(new_data[5081:]))
st.line_chart(day_pred, scaler.inverse_transform(lst_output), 'r')
