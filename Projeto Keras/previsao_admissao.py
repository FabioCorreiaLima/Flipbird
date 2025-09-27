import numpy as np
from keras.models import load_model

# 1. Definir as colunas (features) de entrada
COLUNAS = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]

REGRAS_NEGOCIO = {
    "GRE Score": (260, 340, int),
    "TOEFL Score": (0, 120, int),
    "University Rating": (1, 5, int),
    "SOP": (1, 5, float),
    "LOR": (1, 5, float),
    "CGPA": (0.0, 10.0, float),
    "Research": (0, 1, int)
}

def obter_input_usuario(feature, min_val, max_val, tipo):
    while True:
        try:
            prompt = f"Digite o valor para {feature} (Intervalo: {min_val} a {max_val}, Tipo: {tipo.__name__}): "
            valor_str = input(prompt)
            valor = tipo(valor_str)
            
            if tipo == int:
                if not (min_val <= valor <= max_val):
                    print(f"Erro: O valor deve estar no intervalo de {min_val} a {max_val}.")
                    continue
            elif tipo == float:
                if not (min_val <= valor <= max_val):
                    print(f"Erro: O valor deve estar no intervalo de {min_val} a {max_val}.")
                    continue
            
            if feature == "Research" and valor not in [0, 1]:
                 print("Erro: O valor para Research deve ser 0 (Não) ou 1 (Sim).")
                 continue

            return valor
        except ValueError:
            print(f"Erro: O valor '{valor_str}' não é um {tipo.__name__} válido. Tente novamente.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")

def fazer_previsao():
    
    # 2. Carregar o modelo pré-treinado
    try:
        modelo = load_model('modelo_treinado.keras')
        print("---")
        print("Modelo 'modelo_treinado.keras' carregado com sucesso.")
        print("---")
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo. Verifique se o arquivo 'modelo_treinado.keras' existe e se as bibliotecas (numpy, keras) estão instaladas corretamente.")
        print(f"Detalhes do erro: {e}")
        return

    # 3. Solicitar os valores das características ao usuário
    valores_entrada = []
    print("Por favor, digite os valores para as características do candidato:")
    
    for feature in COLUNAS:
        min_val, max_val, tipo = REGRAS_NEGOCIO.get(feature, (None, None, float))
        if feature == "Research":
            min_val, max_val, tipo = 0, 1, int
        elif feature in ["University Rating", "GRE Score", "TOEFL Score"]:
            min_val, max_val, tipo = REGRAS_NEGOCIO[feature]
        elif feature in ["SOP", "LOR", "CGPA"]:
            min_val, max_val, tipo = REGRAS_NEGOCIO[feature]

        valor = obter_input_usuario(feature, min_val, max_val, tipo)
        valores_entrada.append(valor)
    
    print("\n---")
    print("Valores de entrada coletados:")
    for i in range(len(COLUNAS)):
        print(f"{COLUNAS[i]}: {valores_entrada[i]}")
    print("---")
    
    # 4. Transformar os valores para um array numpy no formato correto
    entrada_array = np.array(valores_entrada).reshape(1, -1)
    
    # 5. Gerar previsão
    print("Gerando previsão...")
    previsao_raw = modelo.predict(entrada_array)

    chance_admissao_decimal = previsao_raw[0][0]
    
    # 6. Exibir a chance prevista de admissão em percentual
    chance_admissao_percentual = chance_admissao_decimal * 100
    
    print("\n-------------------------------------------")
    print(f"Chance de Admissão (Decimal): {chance_admissao_decimal:.4f}")
    print(f"Chance Prevista de Admissão: {chance_admissao_percentual:.2f}%")
    print("-------------------------------------------")

if __name__ == "__main__":
    fazer_previsao()