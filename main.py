from model_train import predict_and_learn

def app() -> None:
    print("👤 Bem-vindo ao classificador de gênero!")
    
    while True:
        print("\nEscolha uma opção:")
        print("1. Fazer uma previsão")
        print("2. Sair")
        choice = input("Opção (1/2): ")
        
        if choice == '1':
            try:
                altura = int(input("Digite a altura (cm): "))
                peso = int(input("Digite o peso (kg): "))
                calcado = int(input("Digite o tamanho do calçado: "))
                genero_real = input("Digite o gênero real (male/female) ou deixe vazio para pular: ").strip()
                
                genero_real = genero_real if genero_real in ['male', 'female'] else None
                
                predict_and_learn(altura, peso, calcado, genero_real)
            except ValueError:
                print("❌ Entrada inválida. Por favor, insira valores numéricos para altura, peso e calçado.")
            except Exception as e:
                print(f"❌ Ocorreu um erro: {e}")
        elif choice == '2':
            print("👋 Saindo...")
            break
        else:
            print("❌ Opção inválida. Tente novamente.")

if __name__ == '__main__':
    app()