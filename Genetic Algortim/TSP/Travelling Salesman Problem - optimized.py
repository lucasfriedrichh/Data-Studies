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

def cruzamento_ox(pais):
    tamanho = len(pais[0])
    filho = [-1]*tamanho
    inicio, fim = sorted(random.sample(range(tamanho), 2))
    filho[inicio:fim] = pais[0][inicio:fim]
    restante = [cidade for cidade in pais[1] if cidade not in filho]
    filho = filho[:inicio] + restante[:tamanho-fim] + filho[inicio:fim] + restante[tamanho-fim:]
    return filho

def mutacao_2opt(rota, matriz_distancias):
    tamanho = len(rota)
    i, j = sorted(random.sample(range(tamanho), 2))
    if i == 0 and j == tamanho - 1:
        return
    custo_antigo = matriz_distancias[rota[i-1]][rota[i]] + matriz_distancias[rota[j]][rota[(j+1)%tamanho]]
    custo_novo = matriz_distancias[rota[i-1]][rota[j]] + matriz_distancias[rota[i]][rota[(j+1)%tamanho]]
    if custo_novo < custo_antigo:
        rota[i:(j+1)] = reversed(rota[i:(j+1)])

def algoritmo_genetico(matriz_distancias, num_geracoes=1000, tamanho_populacao=100, elite_size=10, parada_antecipada=100):
    populacao = gerar_populacao_inicial(len(matriz_distancias), tamanho_populacao)
    melhor_custo = float('inf')
    contador_parada = 0

    for _ in range(num_geracoes):
        pais = selecionar_pais(populacao, matriz_distancias)
        elite = sorted(populacao, key=lambda rota: calcular_custo(rota, matriz_distancias))[:elite_size]
        for _ in range(tamanho_populacao):
            filho = cruzamento_ox(pais)
            mutacao_2opt(filho, matriz_distancias)
            populacao.append(filho)
        populacao.sort(key=lambda rota: calcular_custo(rota, matriz_distancias))
        populacao = elite + populacao[:tamanho_populacao-elite_size]

        custo_atual = calcular_custo(populacao[0], matriz_distancias)
        if custo_atual < melhor_custo:
            melhor_custo = custo_atual
            contador_parada = 0
        else:
            contador_parada += 1

        if contador_parada >= parada_antecipada:
            break

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


melhor_rota, custo = algoritmo_genetico(matriz_distancias, num_geracoes=5000, tamanho_populacao=100, elite_size=20, parada_antecipada=500)
print("Melhor rota: ", melhor_rota)
print("Custo da melhor rota: ", custo)