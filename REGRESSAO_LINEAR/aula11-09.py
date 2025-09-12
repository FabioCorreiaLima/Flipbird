import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.io as pio
pio.renderers.default = "browser"

tabela_historico = pd.read_csv('historico.csv')


y = tabela_historico["Vendas"]
lista_colunas = ["Mês"]
x = tabela_historico[lista_colunas]
modelo = LinearRegression()

modelo.fit(x, y)

meses_prever = pd.DataFrame({"Mês": [25, 26, 27]})
previsoes = modelo.predict(meses_prever)
print(previsoes)
tabela_previsoes = pd.DataFrame(
    {
        "Mês": [25, 26, 27],
        "Vendas": previsoes,
    }
)

tabela_historico["Tipo"] = "Histórico"
tabela_previsoes["Tipo"] = "Previsão"

tabela_grafico = pd.concat([tabela_historico, tabela_previsoes])
print(tabela_grafico)
grafico =px.scatter(
    tabela_grafico,
    x="Mês",
    y="Vendas",
    color="Tipo",
    title="Vendas x Mês",
    labels={"Vendas": "Vendas (R$)", "Mês": "Mês"},
    trendline="ols"
)
grafico.show()



