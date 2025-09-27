import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('admission_dataset.csv')

y = df['Chance of Admit ']
x = df.drop(columns=['Chance of Admit '])

x_treino, x_teste = x[0:300], x[300:]
y_treino, y_teste = y[0:300], y[300:]

modelo = Sequential()
modelo.add(Dense(units=3, activation='relu', input_dim=x_treino.shape[1]))

# camada de saída
modelo.add(Dense(units=1, activation='linear'))

# compilando o modelo
modelo.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])

resultado = modelo.fit(x_treino, y_treino, 
                       epochs=200,
                       batch_size=32,
                       validation_data=(x_teste, y_teste))

plt.plot(resultado.history['loss'], label='loss')
plt.plot(resultado.history['val_loss'], label='val_loss')
plt.title('Histórico de treinamento')
plt.ylabel('Função de custo')
plt.xlabel('Épocas de treinamento')
plt.legend(['Erro de treino', 'Erro de teste'])
plt.show()
plt.savefig('grafico_perda.png')
plt.close()

#salvando o modelo7
modelo.save('modelo_treinamento.keras')