# coding=utf-8

# IMPLEMENTACION DE GMM PARA DATOS MULTIVARIADOS
# Basandonos en The Multivariate Gaussian Distribution y Mixture Models & EM University of Texas at Dalla:
# http://cs229.stanford.edu/section/gaussians.pdf
# Mod-01 Lec-10  : https://www.youtube.com/watch?v=YgExEVji7xs
# https://www.utdallas.edu/~nrr150130/cs7301/2016fa/lects/Lecture_17_GMM.pdf


import pandas as pd  # para trabajo con el IRIS
import seaborn as sns  # para graficos de histogramas de los datos
import numpy as np  # trabajo con matrices
import matplotlib.pyplot as plt  # trabajo con gráficos
from scipy.stats import norm


sns.set_style("white")


# Comenzamos leyendo la base de datos


def panda(path):
    iris = pd.read_csv(path)
    iris = iris.drop('Id', axis=1)
    x = np.array(
        iris.drop(['Species'], 1)).T  # usamos la transpuesta para trabajar con las caracteristicas y no por filas
    y = np.array(iris['Species'])
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  # usamos para cambiar los datos de las imagenes
    for i in range(0, len(y)):
        for j in range(len(labels)):
            if labels[j] == y[i]:
                y[i] = j + 1
    return x, y


X, y = panda("/home/duilmarc/PycharmProjects/sexta_practica/ia/Iris.csv")

# para poder trabajar con los datos usamos DATAFRAMES

data = dict(caracteristica1=X[0], caracteristica2=X[1], caracteristica3=X[2], caracteristica4=X[3])

caract = pd.DataFrame(data=data)

np.random.seed(15)

# la data de iris la clasificaremos en 3 clases
# usaremos una tupla para guardar los vectores de medias , asi las matrices de covarianza, y los pesos
# para cada subclase
# theta representara el mixture model donde theta = { peso1, peso2, peso3, peso4 ... , alfa1, alfa2, ... ,alfa4 }
# donde alfa = a la dupla ( media, matriz de covarianza )


parametros_theta = dict(
    media_subclase_1=[np.random.uniform(max(X[0]), min(X[0])), np.random.uniform(max(X[1]), min(X[1])),
                      np.random.uniform(max(X[2]), min(X[2])), np.random.uniform(max(X[3]), min(X[3]))],
    media_subclase_2=[np.random.uniform(max(X[0]), min(X[0])), np.random.uniform(max(X[1]), min(X[1])),
                      np.random.uniform(max(X[2]), min(X[2])), np.random.uniform(max(X[3]), min(X[3]))],
    media_subclase_3=[np.random.uniform(max(X[0]), min(X[0])), np.random.uniform(max(X[1]), min(X[1])),
                      np.random.uniform(max(X[2]), min(X[2])), np.random.uniform(max(X[3]), min(X[3]))],
    matriz_covarianza_1=[   [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],

    matriz_covarianza_2=[   [0.6, 0, 0, 0],
                            [0, 0.5, 0, 0],
                            [0, 0, 0.5, 0],
                            [0, 0, 0, 1]],

    matriz_covarianza_3=[   [0.5, 0, 0, 0],
                            [0, 0.5, 0, 0],
                            [0, 0, 0.3, 0],
                            [0, 0, 0, 1]],

    pesos=[0.33333333, 0.333333335, 0.333333335])



"""def distribucion_gaussiana_univariable(X, media_x_i, covarianza_x):
    return (1 / np.sqrt(2 * np.pi * covarianza_x)) * np.exp(-(1 / 2) *
     np.transpose(X - media_x_i) * (1 / covarianza_x) * (X - media_x_i))
"""

# De acuerdo a the diagonal covariance matrix case podemos definir a la distribución gaussiana multivariable como una
# multiplicacion de univariables aplicada a cada caracteristica en la posicion que se esta tomando

def distribucion_gaussiana_multivariable(vector_i, vector_media, matriz_covarianza):
    mult = 1.0
    for i in range(len(vector_i)):
        # mult = mult * distribucion_gaussiana_univariable(vector_i[i], vector_media[i], matriz_covarianza[i][i])
        mult = mult * norm.pdf(vector_i[i], vector_media[i], matriz_covarianza[i][i])
    return mult


# calculamos el expectation con nuestros datos de entrada y parametros

""" Apoyandonos de https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f aplicamos las ecuaciones para 
    el expectation y maximitation """

def expectation(valores, parametros_theta):
    gamma = np.zeros((valores.shape[0], 3))
    for vector_i in range(valores.shape[0]):  # recorremos nuestros valores de la base de datos iris
        caracteristicas = [valores['caracteristica1'][vector_i], valores['caracteristica2'][vector_i],
                           valores['caracteristica3'][vector_i],
                           valores['caracteristica4'][vector_i]]
        suma_especie_x_caracteristica = np.zeros(3)
        suma_especie_x_caracteristica[0] = parametros_theta['pesos'][0] * distribucion_gaussiana_multivariable(
            caracteristicas, list(parametros_theta['media_subclase_1']), list(parametros_theta['matriz_covarianza_1']))
        suma_especie_x_caracteristica[1] = parametros_theta['pesos'][1] * distribucion_gaussiana_multivariable(
            caracteristicas, list(parametros_theta['media_subclase_2']), list(parametros_theta['matriz_covarianza_2']))
        suma_especie_x_caracteristica[2] = parametros_theta['pesos'][2] * distribucion_gaussiana_multivariable(
            caracteristicas, list(parametros_theta['media_subclase_3']), list(parametros_theta['matriz_covarianza_3']))

        # Teorema de bayes para gamma para los 3 cluster ( especies ) Posterior Responsibilities using Bayes Rule
        gamma[vector_i, 0] = suma_especie_x_caracteristica[0] / np.sum(
            suma_especie_x_caracteristica)
        gamma[vector_i, 1] = suma_especie_x_caracteristica[1] / np.sum(
            suma_especie_x_caracteristica)
        gamma[vector_i, 2] = suma_especie_x_caracteristica[2] / np.sum(
            suma_especie_x_caracteristica)
    return gamma


gamma = expectation(caract, parametros_theta)

def recalcular_medias(gamma, caract, parametros_theta,N, n_k):

    acumulador_media = np.zeros((N, gamma.shape[1], caract.shape[1]))
    for n in range(N):
        caracteristicas = [caract['caracteristica1'][n], caract['caracteristica2'][n],
                           caract['caracteristica3'][n], caract['caracteristica4'][n]]
        for k in range(gamma.shape[1]):
            escalar = gamma[n][k]
            acumulador_media[n][k][0] = escalar * caracteristicas[0]
            acumulador_media[n][k][1] = escalar * caracteristicas[1]
            acumulador_media[n][k][2] = escalar * caracteristicas[2]
            acumulador_media[n][k][3] = escalar * caracteristicas[3]

    j_k = np.sum(acumulador_media, axis=0)
    parametros_theta['media_subclase_1'] = 1 / n_k[0] * j_k[0]
    parametros_theta['media_subclase_2'] = 1 / n_k[1] * j_k[1]
    parametros_theta['media_subclase_3'] = 1 / n_k[2] * j_k[2]

def recalcular_matriz_covarianza(gamma, caract, parametros_theta, n_k):

    for k in range(gamma.shape[1]):
        x = caract - parametros_theta['media_subclase_'+str(k+1)] # restamos x - media nueva generada

        gamma_diago = np.diag(gamma[:,k]) # colocamos la matriz de valores calculados de gama para la columna k en una matriz
                                          # diagonal dimension n x n

        gamma_diago = np.matrix(gamma_diago)
        sigma_k = x.values.T * gamma_diago * x.values  #         ( x-nueva_media )t * gamma_k * ( x- nueva_media )
        parametros_theta['matriz_covarianza_'+str(k+1)] = sigma_k/n_k[k]
        print parametros_theta['matriz_covarianza_'+str(k+1)]



def Maximitation(gamma, caract):
    global parametros_theta

    N= caract.shape[0]
    """ calculamos la suma de todos los valores en cada columna y lo dividimos entre el total de datos """
    n_k = np.sum( gamma, axis = 0 )

    """ Actualizamos los pesos """
    for peso in range(len(parametros_theta['pesos'])):
        parametros_theta['pesos'][peso] = n_k[peso]/N

    """ Actualizamos las medias"""
    recalcular_medias(gamma, caract,parametros_theta,N,n_k)

    """ Actualizar matriz de covarianza"""
    recalcular_matriz_covarianza(gamma, caract,parametros_theta,n_k)






""" 
sns.distplot(X[0], bins=20, kde=False)
sns.distplot(X[1], bins=20, kde=False)
sns.distplot(X[2], bins=20, kde=False)
sns.distplot(X[3], bins=20, kde=False)

2"""
sns.set_color_codes()
sns.distplot(X[0],  kde_kws={"color": "k", "lw": 3, "label": "caracteristica1"}, hist_kws={"histtype": "step", "linewidth": 3})
sns.distplot(X[1],  kde_kws={"color": "g", "lw": 3, "label": "caracteristica2"}, hist_kws={"histtype": "step", "linewidth": 3})
sns.distplot(X[2],  kde_kws={"color": "b", "lw": 3, "label": "caracteristica3"}, hist_kws={"histtype": "step", "linewidth": 3})
sns.distplot(X[3],   kde_kws={"color": "r", "lw": 3, "label": "caracteristica4"}, hist_kws={"histtype": "step", "linewidth": 3})

plt.show()
