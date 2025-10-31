import pandas as pd

from sklearn import tree

# 1 - Import da base de dados
dados = pd.read_csv('brasil_pessoas.csv')

x_train = dados[['altura', 'peso', 'calçado']].values # Valores de entrada
y_train = dados['gênero'].values # Valores de saída

# 2 - Treinamento da arvore de decisão
clf = tree.DecisionTreeClassifier() # Criando arvore de decisão
clf = clf.fit(x_train, y_train) # Treinando a arvore de decisão com os dados

def get_model() -> tree.DecisionTreeClassifier:
    """
    Retorna o modelo treinado de acordo com os dados do "brasil_pessoas.csv".

    Returns:
        tree.DecisionTreeClassifier: Modelo treinado.
    """
    return clf

if __name__ == '__main__':
    pass