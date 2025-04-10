import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)

try:
    df = pd.read_csv('clientes.csv')
    print("Base de dados importada com sucesso!")
    print("\nPrimeiras linhas da tabela original:")
    print(df.head())

    # Inicializa o LabelEncoder
    label_encoder = LabelEncoder()

    # Aplica o Label Encoding a cada coluna especificada
    colunas_para_codificar = ['profissao', 'mix_credito', 'comportamento_pagamento']
    for coluna in colunas_para_codificar:
        if coluna in df.columns:
            df[coluna] = label_encoder.fit_transform(df[coluna])
            print(f"\nColuna '{coluna}' após Label Encoding:")
            print(df[[coluna]].head())
        else:
            print(f"\nColuna '{coluna}' não encontrada no DataFrame.")

    print("\nPrimeiras linhas da tabela após Label Encoding:")
    print(df.head())
    print("\nInformações sobre os tipos de dados após Label Encoding:")
    print(df.info())

    # Supondo que seu DataFrame se chama 'df' e a coluna alvo é 'score_credito'
    y = df['score_credito']  # Seleciona a coluna 'score_credito' para ser a variável alvo (y)
    X = df.drop('score_credito', axis=1)  # Seleciona todas as outras colunas para serem as features (X)


    print("\nVariável alvo (y) - Primeiras linhas:")
    print(y.head())
    print("\nFeatures (X) - Primeiras linhas:")
    print(X.head())

    # Separa os dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("\nTamanho do conjunto de treino de X:", X_train.shape)
    print("Tamanho do conjunto de teste de X:", X_test.shape)
    print("Tamanho do conjunto de treino de y:", y_train.shape)
    print("Tamanho do conjunto de teste de y:", y_test.shape)

    # **1. Importar o modelo** (já feito nas importações)

    # **2. Criar o modelo base**
    rf_base = RandomForestRegressor(random_state=42)

    # **3. Treinar o modelo base**
    rf_base.fit(X_train, y_train)
    print("\nModelo Random Forest base treinado!")

    # **4. Ajuste de Hiperparâmetros com GridSearchCV**
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Obtém os melhores hiperparâmetros encontrados
    best_params = grid_search.best_params_
    print(f"\nMelhores hiperparâmetros encontrados: {best_params}")

    # Obtém o melhor modelo treinado
    best_model = grid_search.best_estimator_

    # **5. Fazer Previsões com o Melhor Modelo**
    y_pred_best = best_model.predict(X_test)
    
    # Obtém o melhor modelo treinado
    best_model = grid_search.best_estimator_
    print("\nMelhor modelo encontrado e treinado para previsões.")

    # **SALVAR O MODELO TREINADO**
    nome_arquivo_modelo = 'modelo_treinado.pkl'
    with open(nome_arquivo_modelo, 'wb') as arquivo_pkl:
        pickle.dump(best_model, arquivo_pkl)

    print(f"\nModelo treinado salvo com sucesso em '{nome_arquivo_modelo}'")

    print("\nPrimeiras 10 previsões do melhor modelo no conjunto de teste:")
    print(y_pred_best[:10])
    print("\nPrimeiros 10 valores reais da nota de crédito no conjunto de teste:")
    print(y_test[:10].values)

    # **6. Avaliar o Melhor Modelo**
    mae_best = mean_absolute_error(y_test, y_pred_best)
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    r2_best = r2_score(y_test, y_pred_best)

    print(f"\nDesempenho do Melhor Modelo no Conjunto de Teste (após ajuste de hiperparâmetros):")
    print(f"Erro Médio Absoluto (MAE): {mae_best:.2f}")
    print(f"Erro Quadrático Médio (MSE): {mse_best:.2f}")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse_best:.2f}")
    print(f"R-quadrado (R²): {r2_best:.4f}")

except FileNotFoundError:
    print("Erro: O arquivo 'clientes.csv' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro ao importar o arquivo: {e}")