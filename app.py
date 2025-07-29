import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from cria_modelo import SelectFeatures, MinMax

## Funções

def pipeline(df):

    pipeline = Pipeline([
        ('select_features', SelectFeatures()),
        ('min_max', MinMax())
    ])

    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

def separa_treino_teste(df):
    seed = 13
    x, y = df[['Height', 'Weight']], df['Obesity']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)
    return x_train, x_test, y_train, y_test

def prepara_modelo():

    # Importa dados
    df = pd.read_csv('Dados/Obesity.csv', sep =',')

    # Separa nos conjuntos de treino e teste
    x_train, x_test, y_train, y_test = separa_treino_teste(df)

    # Importa modelo
    modelo = joblib.load('Modelo/knn.joblib')

    return x_train, x_test, y_train, y_test, modelo

def valida_inputs(lstInput):

    if len(lstInput) < 2:
        st.warning('Confirme se os campos de "peso" e "altura" foram preenchidos.')
        return False

    altura = lstInput[0]
    peso = lstInput[1]

    if peso == None or altura == None:
        st.warning('Confirme se os campos de "peso" e "altura" foram preenchidos.')
        return False

    if peso == 0:
        st.warning('Informe valor de peso superior a 0.')
        return False

    if peso > 400:
        st.warning('Peso informado está além do range de trabalho. Confirme se valor do peso está em kg.')
        return False

    if altura == 0:
        st.warning('Informe valor de altura superior a 0.')
        return False

    if altura > 3:
        st.warning('Altura informada está além do range de trabalho. Confirme se valor da altura está em metros.')
        return False

    return True

def faz_predicao(x_test, lstInput, modelo):

    # Valida dados inputados
    if valida_inputs(lstInput):

        # Adiciona dados inputados ao conjunto de teste e faz transformações
        x_input = pd.DataFrame([lstInput], columns = x_test.columns)
        x_test = pd.concat([x_test, x_input], axis=0)
        x_test = pipeline(x_test)

        # Faz predição
        y_pred = modelo.predict(x_test)

        return y_pred[-1]

    return None

## Execução do Streamlit

x_train, x_test, y_train, y_test, modelo = prepara_modelo()

st.write('# Avaliador')
st.write('Esta ferramenta faz uma avaliação simplificada, e é voltada para identificação de quadros de obesidade.')
st.write('Para uma avaliação mais precisa, consulte um médico.')

st.divider()

# Pega inputs
st.write('### Dados da pessoa')

col1, col2 = st.columns(2)

with col1:
    st.write('#### Peso')
    peso = st.number_input('Qual o peso (em kg)?', min_value = 0.0, format = '%0.1f', step = 0.1)
    st.write(f'Peso informado (kg): {peso}')

with col2:
    st.write('#### Altura')
    altura = st.number_input('Qual a altura (em metros)?', min_value = 0.0, format = '%0.2f', step = 0.1)
    st.write(f'Altura informada (m): {altura}')

if st.button('Avaliar'):

    resultado = faz_predicao(x_test, [altura, peso], modelo)

    if resultado != None:
        st.success(f'Avaliação feita! Quadro identificado como: **{resultado}**')

