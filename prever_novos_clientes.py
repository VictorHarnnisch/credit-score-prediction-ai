import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Carregar o modelo treinado
with open('modelo_treinado.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Carregar os novos dados
try:
    novos_clientes_df = pd.read_csv('novos_clientes.csv')
    print("Base de dados de novos clientes importada com sucesso!")

    # Inicializar o LabelEncoder (você precisaria garantir que ele seja consistente
    # com o usado no treinamento - idealmente, carregar o encoder salvo também)
    label_encoder = LabelEncoder()
    colunas_para_codificar = ['profissao', 'mix_credito', 'comportamento_pagamento']

    # Pré-processar as colunas usando o mesmo encoder
    for coluna in colunas_para_codificar:
        if coluna in novos_clientes_df.columns:
            # Aqui, você idealmente carregaria o encoder treinado e usaria .transform()
            # Para simplificar, estamos criando um novo e ajustando com todos os dados vistos.
            # A forma correta seria carregar o encoder salvo.
            full_data = pd.concat([pd.read_csv('clientes.csv')[coluna], novos_clientes_df[coluna]], ignore_index=True)
            temp_encoder = LabelEncoder().fit(full_data)
            novos_clientes_df[coluna] = temp_encoder.transform(novos_clientes_df[coluna])
            print(f"Coluna '{coluna}' nos novos clientes codificada.")
        else:
            print(f"A coluna '{coluna}' não foi encontrada nos novos clientes.")

    # Selecionar as features (certifique-se de usar as mesmas colunas do X_train)
    features_usadas = ['profissao', 'mix_credito', 'comportamento_pagamento', ...] # Adicione todas as colunas de X_train
    X_novos = novos_clientes_df[features_usadas]
    print("\nFeatures dos novos clientes preparadas.")

    # Fazer as novas previsões
    novas_previsoes = best_model.predict(X_novos)
    print("\nNovas previsões de nota de crédito:")
    print(novas_previsoes)

except FileNotFoundError as e:
    print(f"Erro ao carregar arquivo: {e.filename} não encontrado.")
except Exception as e:
    print(f"Ocorreu um erro ao processar os novos clientes: {e}")