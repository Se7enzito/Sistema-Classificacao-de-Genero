import pandas as pd
import joblib
import os

from sklearn import tree


DATA_PATH = 'brasil_pessoas.csv'
MODEL_PATH = 'modelo_arvore_decisao.pkl'

def train_model() -> tree.DecisionTreeClassifier:
    """Treina o modelo a partir dos dados atuais e o salva."""
    dados = pd.read_csv(DATA_PATH)
    x_train = dados[['altura', 'peso', 'calçado']].values
    y_train = dados['gênero'].values

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    joblib.dump(clf, MODEL_PATH)
    return clf

def get_model() -> tree.DecisionTreeClassifier:
    """Carrega o modelo salvo, treinando novamente se não existir."""
    if not os.path.exists(MODEL_PATH):
        return train_model()
    return joblib.load(MODEL_PATH)

def predict_and_learn(altura: int, peso: int, calcado: int, genero_real: str = None):
    """
    Faz uma previsão e (opcionalmente) atualiza o modelo com o resultado real.

    Args:
        altura (int): Altura em cm.
        peso (int): Peso em kg.
        calcado (int): Tamanho do calçado.
        genero_real (str, opcional): Gênero real, se conhecido ("male" ou "female").
    """
    clf = get_model()

    pred = clf.predict([[altura, peso, calcado]])[0]
    print(f"\n🔮 Previsão: {'Masculino' if pred == 'male' else 'Feminino'}")

    if genero_real:
        print(f"📈 Adicionando novo dado e reentreinando modelo...")

        novo_dado = pd.DataFrame([[altura, peso, calcado, genero_real]],
                                 columns=['altura', 'peso', 'calçado', 'gênero'])
        dados = pd.read_csv(DATA_PATH)
        dados = pd.concat([dados, novo_dado], ignore_index=True)
        dados.to_csv(DATA_PATH, index=False)

        train_model()
        print("✅ Modelo atualizado e salvo.")

    return pred

if __name__ == '__main__':
    pass