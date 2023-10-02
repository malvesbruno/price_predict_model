#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import pathlib 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import time
import os
from geopy.geocoders import Nominatim


# In[83]:


locator = Nominatim(user_agent="myGeocoder")

location = locator.geocode('Jardim Reimberg, São Paulo')

longitude = location.longitude
latitude = location.latitude
print(latitude, longitude)


# In[84]:


base_path = pathlib.Path('DataBase')

for arquivo in base_path.iterdir():
    arquivo = os.path.join(base_path, arquivo.name)
    print(arquivo)
    df = pd.read_csv(arquivo,  sep=',')
display(df)


# In[85]:


df.head(1000).to_excel("Primeiro_registros.xlsx")


# In[86]:


display(df)


# In[87]:


print(df.isnull().sum())


# In[88]:


df = df.drop(['adm-fees'], axis=1) 
df = df.dropna()
print(df.isnull().sum())


# In[89]:


print(df.info())
print("__" * 30)
print(df.iloc[0])


# In[90]:


for index, linha in enumerate(df['garage-places']):
    linha = linha[linha.find('-') + 1: ]
    linha = linha.replace('-', '0')
    df['garage-places'].iloc[index] = linha
    
for index, linha in enumerate(df['rooms']):
    linha = linha[linha.find('-') + 1: ]
    linha = linha.replace('-', '0')
    df['rooms'].iloc[index] = linha

for index, linha in enumerate(df['square-foot']):
    linha = linha[linha.find('-') + 1: ]
    linha = linha.replace('-', '0')
    df['square-foot'].iloc[index] = linha

df['garage-places'] = df['garage-places'].astype(np.float64, copy=False)
df['rooms'] = df['rooms'].astype(np.float64, copy=False)
df['square-foot'] = df['square-foot'].astype(np.float64, copy=False)

display(df)


# In[91]:


plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot = True, cmap = 'Greens')


# In[92]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - (1.5 * amplitude), q3 + (1.5 * amplitude)
def excluir_outliers(df, coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[coluna])
    df = df.loc[(df[coluna] >= lim_inf) & (df[coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


# In[93]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)
    
def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# In[94]:


diagrama_caixa(df['price'])
histograma(df['price'])


# In[95]:


df, qtde_linhas = excluir_outliers(df, 'price')
print(f'{qtde_linhas} removidas na coluna de price')
histograma(df['price'])


# In[96]:


diagrama_caixa(df['garage-places'])
histograma(df['garage-places'])


# In[97]:


df, qtde_linhas = excluir_outliers(df, 'garage-places')
print(f'{qtde_linhas} removidas na coluna de garage-places')
histograma(df['garage-places'])


# In[98]:


diagrama_caixa(df['rooms'])
histograma(df['rooms'])


# In[99]:


df, qtde_linhas = excluir_outliers(df, 'rooms')
print(f'{qtde_linhas} removidas na coluna de rooms')
histograma(df['rooms'])


# In[100]:


diagrama_caixa(df['square-foot'])
histograma(df['square-foot'])


# In[101]:


df, qtde_linhas = excluir_outliers(df, 'square-foot')
print(f'{qtde_linhas} removidas na coluna de square-foot')
histograma(df['square-foot'])


# In[102]:


plt.figure(figsize = (15, 5))
df = df.drop(['address', 'city', 'neighborhood'], axis=1)
print(df.shape)
print(df.info())


# In[ ]:





# In[103]:


display(df)
amostra = df.sample(n=500)


centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}

mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price',zoom=10,
                         center=centro_mapa, mapbox_style='stamen-terrain', radius=2.5)

mapa.show()


# In[104]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}\nR²:{r2}\nRSME:{RSME}\n'


# In[105]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
           'LinearRegression': modelo_lr,
           'ExtraTrees': modelo_et,
          }

X= df.drop('price', axis=1)
y= df['price']


# In[106]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar modelo
    modelo.fit(X_train, y_train)
    #teste modelo
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# In[107]:


importancia_features = pd.DataFrame(data=modelo_et.feature_importances_, index=X_test.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)

plt.figure(figsize=(15,5))
grafico = sns.barplot(x=importancia_features.index, y=importancia_features[0])
grafico.tick_params(axis='x', rotation= 90)
display(df)


# In[108]:


display(df)


# In[109]:


X= df.drop('price', axis=1)
y= df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar modelo
    modelo.fit(X_train, y_train)
    #teste modelo
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# In[111]:


print(df.columns)


# In[112]:


import joblib
joblib.dump(modelo_et, 'modelo.joblib')


# In[ ]:




