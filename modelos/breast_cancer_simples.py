import pandas as pd
from sklearn.model_selection import train_test_split

previsores = pd.read_csv(f'../datasets/breast_cancer/entradas_breast.csv')
classe = pd.read_csv('../datasets/breast_cancer/saidas_breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe,
                                                                                              test_size=0.25)
"""
Sequential cria uma sequencia de camadas, de entrada, oculta e outras.
"""
from keras.models import Sequential

"""Dense é utilizada em camadas densas, onde cada um dos neurônios é ligado a todos os outros da próxima camada
Também chamada de rede neural Fully Connected
"""
from keras.layers import Dense

"""Criando uma rede neural"""
classificador = Sequential()
"""Adicionando primeira camada oculta"""
"""units: Quantidade de quantidade de neurônios = 
(número de atributos previsores + número de neurônios da camada de saída)/2"""

classificador.add(
    Dense(16, activation='relu', kernel_initializer='random_uniform', input_dim=30)
)
classificador.add(
    Dense(units=1, activation='sigmoid')
)

