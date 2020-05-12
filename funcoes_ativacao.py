import numpy as np


# Retorna 0 ou 1
def step_function(soma):
    return 1 if soma >= 1 else 0


# Retorna valores entre 0 e 1
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


# Retorna valores entre -1 e +1
def tanh_function(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0


def linearFunction(soma):
    return soma


def softmaxFunction(vetor):
    ex = np.exp(vetor)
    return ex / ex.sum()


print(step_function(-1))
print(step_function(2))
print(sigmoid_function(0.358))
print(tanh_function(-0.358))
print(reluFunction(0.358))
print(linearFunction(-0.358))
valores = [5.0, 2.0, 1.3]
print(softmaxFunction(valores))

print("-------------------------")

