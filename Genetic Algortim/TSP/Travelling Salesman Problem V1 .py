# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:53:03 2024

@author: Lucas Friedrich
"""

import numpy as np
import random

def calcular_custo(rota, matriz_distancias):
    custo = 0
    for i in range(len(rota)-1):
        custo += matriz_distancias[rota[i]][rota[i+1]]
    custo += matriz_distancias[rota[-1]][rota[0]]
    return custo

def gerar_populacao_inicial(num_cidades, tamanho_populacao):
    populacao = []
    for _ in range(tamanho_populacao):
        rota = list(range(num_cidades))
        random.shuffle(rota)
        populacao.append(rota)
    return populacao

def selecionar_pais(populacao, matriz_distancias):
    custos = [calcular_custo(rota, matriz_distancias) for rota in populacao]
    indices = list(range(len(populacao)))
    indices.sort(key=lambda x: custos[x])
    return [populacao[i] for i in indices[:2]]

def cruzamento(pais):
    corte = random.randint(0, len(pais[0])-1)
    filho = pais[0][:corte] + [cidade for cidade in pais[1] if cidade not in pais[0][:corte]]
    return filho

def mutacao(rota):
    i, j = random.sample(range(len(rota)), 2)
    rota[i], rota[j] = rota[j], rota[i]

def algoritmo_genetico(matriz_distancias, num_geracoes=1000, tamanho_populacao=100):
    populacao = gerar_populacao_inicial(len(matriz_distancias), tamanho_populacao)
    for _ in range(num_geracoes):
        pais = selecionar_pais(populacao, matriz_distancias)
        for _ in range(tamanho_populacao):
            filho = cruzamento(pais)
            mutacao(filho)
            populacao.append(filho)
        populacao.sort(key=lambda rota: calcular_custo(rota, matriz_distancias))
        populacao = populacao[:tamanho_populacao]
    return populacao[0], calcular_custo(populacao[0], matriz_distancias)


matriz_distancias = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 15],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 10],
    [25, 15, 20, 10, 0]
]

"""
num_cidades = 5
np.random.seed(0)  # para garantir que os resultados sejam reproduzíveis
matriz_distancias = np.random.randint(1, 100, size=(num_cidades, num_cidades))
np.fill_diagonal(matriz_distancias, 0)

print("Matriz de distâncias:")
print(matriz_distancias)
"""

melhor_rota, custo = algoritmo_genetico(matriz_distancias)
print("Melhor rota: ", melhor_rota)
print("Custo da melhor rota: ", custo)
