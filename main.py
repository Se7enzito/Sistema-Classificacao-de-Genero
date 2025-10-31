from model_train import get_model

# 1 - Pegando o modelo
clf = get_model()

# 2 - Solicitando dados do usuário
altura = float(input("Digite a altura (em metros): "))
peso = float(input("Digite o peso (em kg): "))
calcado = int(input("Digite o tamanho do calçado: "))

# 3 - Fazendo a predição
entrada = [[altura, peso, calcado]]
previsao = clf.predict(entrada)

# 4 - Tratando e exibindo o resultado
if previsao[0] == 'male':
    genero = 'Masculino'
else:
    genero = 'Feminino'

print(f"O gênero previsto é: {genero}")