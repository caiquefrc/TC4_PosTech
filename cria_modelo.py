# Importa bibliotecas
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Classe para seleção das variáveis preditoras e target
class SelectFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features_list = ['Height', 'Weight']):
        self.features_list = features_list

    def fit(self, df):
        return self

    def transform(self, df):
        df = df[self.features_list]
        return df

# Classe para normalizar variáveis preditoras
class MinMax(BaseEstimator, TransformerMixin):

    def __init__(self, min_max_features = ['Height', 'Weight']):
        self.min_max_features = min_max_features

    def fit(self, df):
        return self

    def transform(self, df):
        scaler = MinMaxScaler()
        df[self.min_max_features] = scaler.fit_transform(df[self.min_max_features])
        return df

# Função do Pipeline
def pipeline(df):

    pipeline = Pipeline([
        ('select_features', SelectFeatures()),
        ('min_max', MinMax())
    ])

    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

# Função para seleção dos hiperparâmetros do modelo
def get_best_hyperparams(model, params, x_train, y_train):

  grid = GridSearchCV(
      model,
      param_grid = params,
      cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 13)
      )

  grid.fit(x_train, y_train)

  print(f'Melhores Parâmatros: {grid.best_estimator_}')

  return grid.best_estimator_

# Função para avaliar desempenho do modelo
def avaliaModelo(y_pred, y_test):

  acuracia = accuracy_score(y_test, y_pred)
  print(f'Acurácia: {np.round(acuracia*100,2)}%')
  score = f1_score(y_test, y_pred, average = 'weighted')
  print(f'f1_score: {np.round(score*100,2)}%')
  print(classification_report(y_test, y_pred))


###

# Importação dos dados
df = pd.read_csv('Dados/Obesity.csv', sep =',')

# Agrupamento dados da variável alvo
dict_obesity = {
    'Insufficient_Weight': 'abaixo',
    'Normal_Weight': 'normal',
    'Overweight_Level_I': 'sobrepeso',
    'Overweight_Level_II': 'sobrepeso',
    'Obesity_Type_I': 'obesidade',
    'Obesity_Type_II': 'obesidade',
    'Obesity_Type_III': 'obesidade'
}
df['Obesity'] = df['Obesity'].map(dict_obesity)

# Separação dos dados nos conjuntos de treino e teste
seed = 13
x, y = df[['Height', 'Weight']], df['Obesity']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = seed, stratify = y)

# Transformação dos dados com as operações no pipeline
x_train = pipeline(x_train)
x_test = pipeline(x_test)

# Criação do modelo
knn_param_grid = {'n_neighbors': range(1, 20)}
knn_best_params = get_best_hyperparams(KNeighborsClassifier(), knn_param_grid, x_train, y_train)
knn = KNeighborsClassifier(n_neighbors = knn_best_params.n_neighbors).fit(x_train, y_train)

# Faz predição do conjunto de teste
y_pred_knn = knn.predict(x_test)
avaliaModelo(y_pred_knn, y_test)

# Salva o modelo
joblib.dump(knn, 'Modelo/knn.joblib')