# credit-score-prediction-ai
🚀 Projeto de Previsão de Nota de Crédito com Inteligência Artificial 🚀
<div style="display: inline_block"><br/>
  <img align="center" alt="Vit-Python" height="30" width="40" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg">Python<br/>
  <img align="center" alt="Vit-Pandas" height="30" width="40" src="https://github.com/user-attachments/assets/2cf83dfe-59ae-429b-9140-7ed1a2db8062">Pandas<br/>
  <img align="center" alt="Vit-Python" height="30" width="40" src="https://github.com/user-attachments/assets/9439ffe3-a54c-4d8f-be19-b8cfb03a6bcd" width="100">Scikit-Learn<br/>
   <img align="center" alt="Vit-Python" height="30" width="40" src="https://github.com/user-attachments/assets/361b5555-eb35-46ae-aefc-0aa23d5b1afd">Github Stars
  </div><br/>

🎯 Visão Geral do Projeto
Este projeto tem como objetivo desenvolver um modelo de Inteligência Artificial capaz de prever a nota de crédito de clientes com base em um conjunto de dados. O processo envolve desde a compreensão do desafio de negócio até a implementação de um modelo preditivo e a sua aplicação em novos dados.

⚙️ Passo a Passo do Processo
A seguir, detalhamos cada etapa do desenvolvimento deste projeto:

1. 🧐 Entender o Desafio da Empresa
Objetivo: Compreender as necessidades da empresa em relação à previsão de notas de crédito. Isso inclui identificar os objetivos de negócio, as métricas de sucesso esperadas e como a previsão será utilizada (por exemplo, para avaliação de risco, concessão de crédito, etc.).
Considerações: Análise do impacto de uma previsão precisa na tomada de decisões e na otimização de processos.
2. 💾 Importar a Base de Dados
Ação: Carregamento dos dados brutos em um ambiente de análise (Python com a biblioteca Pandas).
Código (Exemplo):
Python

import pandas as pd

try:
    df = pd.read_csv('clientes.csv')
    print("✅ Base de dados importada com sucesso!")
    # Exibir as primeiras linhas para inspeção inicial
    print(df.head())
except FileNotFoundError:
    print("❌ Erro: O arquivo 'clientes.csv' não foi encontrado.")
except Exception as e:
    print(f"⚠️ Ocorreu um erro ao importar o arquivo: {e}")
3. 🧹 Preparar a Base de Dados para Inteligência Artificial
Ações:
Exploração Inicial: Análise das colunas, tipos de dados, valores faltantes e estatísticas descritivas.
Limpeza de Dados: Tratamento de valores ausentes, outliers e inconsistências.
Codificação de Variáveis Categóricas: Conversão de colunas do tipo object (string) para representações numéricas que os modelos de IA possam entender. Neste projeto, utilizamos o LabelEncoder para as colunas: profissao, mix_credito e comportamento_pagamento.
Código (Exemplo de Label Encoding):
Python

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
colunas_para_codificar = ['profissao', 'mix_credito', 'comportamento_pagamento']
for coluna in colunas_para_codificar:
    if coluna in df.columns:
        df[coluna] = label_encoder.fit_transform(df[coluna])
        print(f"➡️ Coluna '{coluna}' codificada com Label Encoding.")
    else:
        print(f"⚠️ Coluna '{coluna}' não encontrada.")
Separação de Features e Variável Alvo: Definição de X (features/variáveis preditoras) e y (variável alvo - score_credito).
Divisão dos Dados: Criação dos conjuntos de treinamento e teste para avaliar o modelo.
4. 🧠 Criar o Modelo de IA
Ação: Escolha e implementação de um algoritmo de aprendizado de máquina para a tarefa de previsão. Neste projeto, utilizamos o modelo de Random Forest Regressor.
Código (Exemplo de Criação e Treinamento):
Python

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Separar X e y (já realizado no passo anterior)
y = df['score_credito']
X = df.drop('score_credito', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo Random Forest
rf_base = RandomForestRegressor(random_state=42)
rf_base.fit(X_train, y_train)
print("✅ Modelo Random Forest treinado!")
5. 🏆 Escolher o Melhor Modelo
Ação: Avaliação do desempenho do modelo utilizando métricas relevantes (para regressão: Erro Médio Absoluto - MAE, Erro Quadrático Médio - MSE, Raiz do Erro Quadrático Médio - RMSE, R-quadrado - R²). Opcionalmente, realizar o ajuste de hiperparâmetros para otimizar o modelo utilizando técnicas como GridSearchCV.

Código (Exemplo de Avaliação):

Python

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred_base = rf_base.predict(X_test)
mae_base = mean_absolute_error(y_test, y_pred_base)
mse_base = mean_squared_error(y_test, y_pred_base)
rmse_base = np.sqrt(mse_base)
r2_base = r2_score(y_test, y_pred_base)

print("\n📊 Desempenho do Modelo Base:")
print(f"MAE: {mae_base:.2f}")
print(f"MSE: {mse_base:.2f}")
print(f"RMSE: {rmse_base:.2f}")
print(f"R²: {r2_base:.4f}")
Ajuste de Hiperparâmetros (Exemplo com GridSearchCV):
 ```python
from sklearn.model_selection import GridSearchCV

param_grid = {
'n_estimators': [100, 200, 300],
'max_depth': [None, 5, 10, 15],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"\n✨ Melhores hiperparâmetros encontrados: {grid_search.best_params_}")   


6. 🔮 Fazer Novas Previsões
Ação: Utilizar o modelo treinado (o melhor modelo encontrado) para prever a nota de crédito de novos clientes a partir de uma nova base de dados (novos_clientes.csv). É crucial aplicar o mesmo pré-processamento (incluindo o mesmo LabelEncoder treinado) aos novos dados.
Código (Exemplo de Previsão em Novos Dados):
Python

import pandas as pd
import pickle

# Carregar o modelo treinado
with open('modelo_treinado.pkl', 'rb') as file:
    best_model = pickle.load(file)
print("💾 Modelo treinado carregado com sucesso!")

# Carregar e pré-processar os novos dados
try:
    novos_clientes_df = pd.read_csv('novos_clientes.csv')
    print("✅ Base de dados de novos clientes importada!")
    # Aplicar o mesmo Label Encoding (idealmente carregando o encoder salvo)
    label_encoder_novos = LabelEncoder()
    colunas_para_codificar = ['profissao', 'mix_credito', 'comportamento_pagamento']
    for coluna in colunas_para_codificar:
        if coluna in novos_clientes_df.columns:
            # Ajustar o encoder com todos os dados vistos para garantir consistência
            full_data = pd.concat([pd.read_csv('clientes.csv')[coluna], novos_clientes_df[coluna]], ignore_index=True)
            label_encoder_novos.fit(full_data)
            novos_clientes_df[coluna] = label_encoder_novos.transform(novos_clientes_df[coluna])
        else:
            print(f"⚠️ Coluna '{coluna}' não encontrada nos novos clientes.")

    # Selecionar as features
    X_novos = novos_clientes_df[X_train.columns]

    # Fazer as previsões
    novas_previsoes = best_model.predict(X_novos)
    novos_clientes_df["score_credito_previsto"] = novas_previsoes
    print("\n✨ Novas Previsões de Nota de Crédito:")
    print(novos_clientes_df[['cliente_id', 'score_credito_previsto']].head()) # Assumindo que 'cliente_id' exista

    # Opcional: Salvar os resultados
    # novos_clientes_df.to_csv('novos_clientes_com_previsoes.csv', index=False)

except FileNotFoundError:
    print("❌ Erro: O arquivo 'novos_clientes.csv' não foi encontrado.")
except Exception as e:
    print(f"⚠️ Erro ao processar novos clientes: {e}")
🛠️ Tecnologias Utilizadas
Python: Linguagem de programação principal.
Pandas: Para manipulação e análise de dados tabulares.
Scikit-Learn (sklearn): Biblioteca de aprendizado de máquina para tarefas como pré-processamento, modelagem e avaliação.
NumPy: Para operações numéricas eficientes.
Pickle: Para salvar e carregar modelos treinados.
📂 Estrutura do Projeto (Exemplo)
.
├── clientes.csv                      # Base de dados de treinamento
├── novos_clientes.csv                # Base de dados para novas previsões
├── case_score.py                     # Script principal para treinamento do modelo
├── prever_novos_clientes.py          # Script para fazer previsões em novos dados
├── modelo_treinado.pkl               # Arquivo para salvar o modelo treinado
├── README.md                         # Este arquivo
└── ...                               # Outros arquivos (logs, etc.)
🚀 Próximos Passos (Sugestões)
Implementar um pipeline de dados mais robusto: Automatizar as etapas de pré-processamento e treinamento.
Explorar outros modelos de IA: Comparar o desempenho do Random Forest com outros algoritmos (Gradient Boosting, Redes Neurais, etc.).
Otimização de hiperparâmetros mais avançada: Utilizar técnicas como RandomizedSearchCV ou otimização Bayesiana.
Implementar um sistema de monitoramento do modelo: Acompanhar o desempenho do modelo em produção e retreiná-lo conforme necessário.
Criação de uma interface de usuário (UI): Permitir que usuários interajam com o modelo para obter previsões de forma intuitiva.
🤝 Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues para relatar bugs ou propor melhorias, e enviar pull requests com suas modificações.

📜 Licença
Este projeto está sob a licença MIT.
