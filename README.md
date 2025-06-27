# 🏠 Aplicativo de Previsão de Preços de Imóveis

![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![PyCaret](https://img.shields.io/badge/PyCaret-3.2-orange?style=for-the-badge)

Este projeto apresenta um pipeline completo de Machine Learning para prever preços de imóveis, desde a análise de dados até o deploy de um aplicativo web interativo.

### 🚀 [https://previsaoregressaolinear-sgzhrkcgy26bvbvgmmg3fe.streamlit.app/)

---

### Visão Geral do Projeto

### 🛠️ Tecnologias e Ferramentas Utilizadas

* **Linguagem:** Python
* **Análise e Modelagem:** Pandas, PyCaret
* **Criação do App Web:** Streamlit
* **Deploy:** Streamlit Community Cloud
* **Ambiente de Desenvolvimento:** Google Colab, VS Code

### ✨ Principais Funcionalidades

* Interface interativa para inserir as características de um imóvel.
* Previsão de preço em tempo real utilizando um modelo de Machine Learning treinado.
* Análise exploratória de dados detalhada no notebook do projeto.
* Seleção e tunagem automática de modelos com PyCaret para garantir a melhor performance.


![previsão](https://github.com/AlanDiego-py/Previs-o_Regress-o_Linear/blob/main/casa.png)

### ⚙️ Como Executar Localmente

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/seu-usuario/projeto-previsao-casas.git](https://github.com/seu-usuario/projeto-previsao-casas.git)
    ```
2.  Navegue até a pasta do projeto:
    ```bash
    cd projeto-previsao-casas
    ```
3.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # ou
    .\venv\Scripts\activate   # Windows
    ```
4.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
5.  Execute o aplicativo Streamlit:
    ```bash
    streamlit run app.py
    ```
