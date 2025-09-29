import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import os
import joblib
import json
import shap

# --- CONFIGURAÇÃO GLOBAL ---
CONFIG = {
    'n_samples': 500,  # Reduzido para um resultado mais rápido
    'test_size': 0.2,
    'random_state': 42,
    'output_dir': 'credit_artifacts',
    'top_n_features': 6,
    'target_proportion': 0.18
}


# --- FUNÇÃO DE GERAÇÃO DE DADOS ---
def gerar_dados_credito(n_samples):
    """
    Gera dados sintéticos e garante a proporção de classes alvo através de um
    método de ranking de risco, eliminando a necessidade de loops.
    """
    np.random.seed(CONFIG['random_state'])
    data = {
        'idade': np.random.randint(18, 70, size=n_samples),
        'score_externo': np.random.randint(300, 1000, size=n_samples),
        'renda_mensal': np.random.lognormal(mean=8.5, sigma=0.8, size=n_samples) * 1000,
        'historico_credito_meses': np.random.randint(6, 360, size=n_samples),
        'num_inadimplencias_passadas': np.random.poisson(0.3, size=n_samples),
        'tipo_moradia': np.random.choice(['Própria', 'Alugada', 'Financiada'], size=n_samples, p=[0.45, 0.45, 0.1])
    }
    df = pd.DataFrame(data)
    df['taxa_comprometimento_renda'] = (df['renda_mensal'] * np.random.uniform(0.1, 0.9, size=n_samples) * (
                df['score_externo'] / 1000) ** -1.5) / df['renda_mensal']
    df['taxa_comprometimento_renda'] = df['taxa_comprometimento_renda'].clip(0.05, 1.0)
    risco_base = - (df['score_externo'] - 300) / 70 - np.log1p(df['historico_credito_meses'])
    risco_divida = (df['taxa_comprometimento_renda'] ** 2) * 20 * (1 - np.tanh((df['renda_mensal'] - 1000) / 5000))
    risco_historico = df['num_inadimplencias_passadas'] * 5
    risco_interacao = ((df['idade'] < 25) & (df['taxa_comprometimento_renda'] > 0.5)) * 4
    logit_final = -3.5 + risco_base + risco_divida + risco_historico + risco_interacao
    logit_final += np.random.normal(0, 1.5, size=n_samples)
    prob_mau_pagador = 1 / (1 + np.exp(-logit_final))
    threshold = np.percentile(prob_mau_pagador, 100 * (1 - CONFIG['target_proportion']))
    df['mau_pagador'] = (prob_mau_pagador >= threshold).astype(int)
    proporcao_final = df['mau_pagador'].mean()
    print(f"Dados gerados com sucesso! Proporção de maus pagadores definida em: {proporcao_final:.2%}")
    return df


# --- IDENTIFICAÇÃO DE FEATURES ---
def identificar_features_relevantes(df_train, top_n):
    print("\nIniciando processo de descoberta genuína das features mais relevantes...")
    X_train = df_train.drop('mau_pagador', axis=1)
    y_train = df_train['mau_pagador']
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features), (
    'cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
    X_train_processed = preprocessor.fit_transform(X_train)
    feature_names = numeric_features + preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
    model = Sequential([Input(shape=(X_train_processed.shape[1],)), Dense(64, activation='relu'), Dropout(0.3),
                        Dense(32, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['auc'])
    model.fit(X_train_processed, y_train, epochs=50, batch_size=32, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_auc', mode='max')], verbose=0)
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train_processed, 50))
    shap_values = explainer.shap_values(shap.sample(X_train_processed, 100))
    feature_importance = pd.DataFrame(list(zip(feature_names, np.abs(shap_values[0]).mean(0))),
                                      columns=['feature', 'importance'])
    feature_importance = feature_importance[feature_importance['feature'] != 'score_externo']
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    top_features_raw = feature_importance['feature'].tolist()
    final_features, seen_base_features = [], set()
    for feature in top_features_raw:
        base_feature = feature.rsplit('_', 1)[0]
        original_columns = df_train.columns
        if base_feature in original_columns and base_feature not in seen_base_features:
            final_features.append(base_feature)
            seen_base_features.add(base_feature)
        elif feature in original_columns and feature not in seen_base_features:
            final_features.append(feature)
            seen_base_features.add(feature)
        if len(final_features) == top_n: break
    return final_features


# --- TREINAR E AVALIAR O SCORECARD OTIMIZADO ---
def treinar_e_avaliar_scorecard(df_train, df_test, top_features):
    print("\n--- Construindo e Otimizando o Modelo de Scorecard ---")
    X_train, y_train = df_train[top_features], df_train['mau_pagador']
    X_test, y_test = df_test[top_features], df_test['mau_pagador']
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features), (
    'cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier',
                                                                LogisticRegression(class_weight='balanced',
                                                                                   random_state=CONFIG['random_state'],
                                                                                   solver='liblinear'))])
    param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_scorecard = grid_search.best_estimator_
    print(f"Otimização concluída. Melhor parâmetro 'C' encontrado: {grid_search.best_params_['classifier__C']}")
    print("\n--- Avaliando a Performance do Scorecard em Dados Novos ---")
    y_pred, y_pred_proba = best_scorecard.predict(X_test), best_scorecard.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=['Bom Pagador', 'Mau Pagador']))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    return best_scorecard


# --- ANALISAR A RELEVÂNCIA GERAL DOS FATORES ---
def analisar_relevancia_scorecard(scorecard_model):
    print("\n--- Analisando os Fatores de Decisão do Scorecard (Visão Geral) ---")
    classifier, preprocessor = scorecard_model.named_steps['classifier'], scorecard_model.named_steps['preprocessor']
    feature_names, coefs = preprocessor.get_feature_names_out(), classifier.coef_[0]
    relevance_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
    relevance_df['abs_coef'] = np.abs(relevance_df['coef'])
    risco_df = relevance_df[relevance_df['coef'] > 0].copy()
    if not risco_df.empty:
        risco_df['relevancia_risco'] = risco_df['abs_coef'] / risco_df['abs_coef'].sum()
        print("\n[Fatores que mais AUMENTAM o Risco de Inadimplência (0 a 1)]")
        print(risco_df[['feature', 'relevancia_risco']].sort_values(by='relevancia_risco', ascending=False).to_string(
            index=False))
    protecao_df = relevance_df[relevance_df['coef'] < 0].copy()
    if not protecao_df.empty:
        protecao_df['relevancia_protecao'] = protecao_df['abs_coef'] / protecao_df['abs_coef'].sum()
        print("\n[Fatores que mais DIMINUEM o Risco (Indicam Bom Pagador) (0 a 1)]")
        print(protecao_df[['feature', 'relevancia_protecao']].sort_values(by='relevancia_protecao',
                                                                          ascending=False).to_string(index=False))


# --- FUNÇÃO ATUALIZADA: ANÁLISE DE CASOS INDIVIDUAIS ---
def analisar_casos_individuais(scorecard_model, df_test, top_features):
    """Encontra e explica em detalhes um exemplo de bom e mau pagador."""
    print("\n--- Análise de Casos Individuais: Entendendo o Porquê do Score ---")

    X_test = df_test[top_features]

    # Prever para todo o conjunto de teste
    y_pred_proba = scorecard_model.predict_proba(X_test)[:, 1]
    y_pred = scorecard_model.predict(X_test)

    # CORREÇÃO: Usar pandas Series com o índice de df_test para evitar KeyError
    proba_series = pd.Series(y_pred_proba, index=df_test.index)
    pred_series = pd.Series(y_pred, index=df_test.index)

    # Encontrar o melhor exemplo de um MAU pagador
    true_positives_mask = (df_test['mau_pagador'] == 1) & (pred_series == 1)
    mau_pagadores_corretos = proba_series[true_positives_mask]
    exemplo_mau_pagador = df_test.loc[mau_pagadores_corretos.idxmax()] if not mau_pagadores_corretos.empty else None

    # Encontrar o melhor exemplo de um BOM pagador
    true_negatives_mask = (df_test['mau_pagador'] == 0) & (pred_series == 0)
    bons_pagadores_corretos = proba_series[true_negatives_mask]
    exemplo_bom_pagador = df_test.loc[bons_pagadores_corretos.idxmin()] if not bons_pagadores_corretos.empty else None

    # Função auxiliar para explicar um cliente
    def explicar_cliente(cliente_series, tipo):
        if cliente_series is None:
            print(f"\nNão foi possível encontrar um exemplo claro de {tipo} no conjunto de teste.")
            return

        print(f"\n[Análise de um Exemplo de {tipo.upper()}]")
        cliente_df = pd.DataFrame(cliente_series).T
        prob_mau = scorecard_model.predict_proba(cliente_df[top_features])[0, 1]
        score = (1 - prob_mau) * 1000

        print("Perfil do Cliente:")
        print(cliente_df[top_features].to_string(index=False))
        print(f"Score de Crédito Calculado: {score:.0f}")

        preprocessor = scorecard_model.named_steps['preprocessor']
        classifier = scorecard_model.named_steps['classifier']
        dados_transformados = preprocessor.transform(cliente_df[top_features])
        feature_names = preprocessor.get_feature_names_out()
        coefs = classifier.coef_[0]
        contribuicoes = dados_transformados[0] * coefs
        df_contrib = pd.DataFrame({'feature': feature_names, 'contribuicao_logit': contribuicoes})

        print("\nFatores de maior RELEVÂNCIA para a decisão deste cliente:")
        risco_cliente = df_contrib[df_contrib['contribuicao_logit'] > 0].copy()
        protecao_cliente = df_contrib[df_contrib['contribuicao_logit'] < 0].copy()

        if not risco_cliente.empty:
            total_risco = risco_cliente['contribuicao_logit'].sum()
            risco_cliente['relevancia (0-1)'] = risco_cliente['contribuicao_logit'] / total_risco
            print("  - Fatores de Risco (o que mais pesou CONTRA):")
            print(risco_cliente[['feature', 'relevancia (0-1)']].sort_values(by='relevancia (0-1)',
                                                                             ascending=False).to_string(index=False,
                                                                                                        header=True))

        if not protecao_cliente.empty:
            protecao_cliente['abs_contribuicao'] = abs(protecao_cliente['contribuicao_logit'])
            total_protecao = protecao_cliente['abs_contribuicao'].sum()
            protecao_cliente['relevancia (0-1)'] = protecao_cliente['abs_contribuicao'] / total_protecao
            print("  - Fatores de Proteção (o que mais pesou A FAVOR):")
            print(protecao_cliente[['feature', 'relevancia (0-1)']].sort_values(by='relevancia (0-1)',
                                                                                ascending=False).to_string(index=False,
                                                                                                           header=True))

    explicar_cliente(exemplo_mau_pagador, "Mau Pagador")
    explicar_cliente(exemplo_bom_pagador, "Bom Pagador")


# --- ORQUESTRADOR PRINCIPAL ---
def main():
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    df_credito = gerar_dados_credito(n_samples=CONFIG['n_samples'])
    df_train, df_test = train_test_split(df_credito, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'],
                                         stratify=df_credito['mau_pagador'])
    top_features = identificar_features_relevantes(df_train, top_n=CONFIG['top_n_features'])
    print(f"\nAs {CONFIG['top_n_features']} features mais importantes foram descobertas: {top_features}")
    scorecard = treinar_e_avaliar_scorecard(df_train, df_test, top_features)
    analisar_relevancia_scorecard(scorecard)
    analisar_casos_individuais(scorecard, df_test, top_features)
    print("\n\nAnálise de crédito concluída com sucesso!")


if __name__ == "__main__":
    main()
