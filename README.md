# 🦖 Dino AI — Reinforcement Learning com Visão Computacional

![Status do Projeto](https://img.shields.io/badge/status-concluído-green)

Este projeto implementa um agente de Reinforcement Learning (DQN) que aprende a jogar o jogo do dinossauro do Chrome utilizando captura de tela, processamento de imagem e OCR. 

O ambiente é customizado usando Gymnasium, e o treinamento é feito com Stable-Baselines3.

## Ações do Agente

0	| Pular

1	| Abaixar

2	| Não fazer nada


## Observação do Ambiente

- Captura de tela da região do jogo -> Conversão para escala de cinza - > Redimensionamento para 83×100 -> Formato final: (83, 100, 1) — uint8

## Requisitos do Sistema

#### 1️⃣ Tesseract OCR (obrigatório)

Instale o Tesseract OCR:

🔗 https://github.com/tesseract-ocr/tesseract

Após a instalação, ajuste o caminho no código:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#### 2️⃣ Jogo do Dino

Abra o Chrome

Acesse chrome://dino

Deixe o jogo visível na tela

Não mova a janela durante o treino

⚠️ As coordenadas da tela estão fixas no código.

##  Como Executar
###  Criar ambiente virtual

python -m venv .venv

.venv\Scripts\activate

###  Instalar dependências

pip install -r requirements.txt

###  Rodar 

python Dino.py



