import pyautogui
import time
import mss
import numpy as np
import cv2
import torch
import torch.nn as nn
import random
from collections import deque


FRAME_SKIP = 4

# ---------------------------------****
# Configurações gerais
# --------------------------------****

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
LR = 1e-4
EPSILON = 1.0

# ----------------------------
# analise do monitor
# --------------------------


def dimensoes_tela():
    print("POSICIONE NO CANTO SUPERIOR ESQUERDO")
    time.sleep(5)
    x1 = pyautogui.position()[0]
    y1 = pyautogui.position()[1]

    print("POSICIONE NO CANTO INFERIOR DIREITO")
    time.sleep(5)
    x2 = pyautogui.position()[0]
    y2 = pyautogui.position()[1]

    monitor = {"top": y1, "left": x1, "height": (y2 - y1), "width": (x2 - x1)}

    return (monitor)


monitor = dimensoes_tela()

# ---------------------------------
# Captura e tratamento de imagem (state)
# --------------------------------


def captura(monitor, n_frames=4):

    conjunto_imagens = []

    with mss.mss() as sct:
        for _ in range(n_frames):
            captura = sct.grab(monitor)  # captura a tela
            # transforma em vetor (altura (pixels), largura(pixels), canais de cor)
            captura_vetor = np.array(captura)
            # sair de BGRA para Grayscale
            captura_gray = cv2.cvtColor(captura_vetor, cv2.COLOR_BGRA2GRAY)

            # redimensiona
            imagem = cv2.resize(captura_gray, (84, 84),
                                interpolation=cv2.INTER_AREA)
            imagem_normalizada = imagem / 255.0

            conjunto_imagens.append(imagem_normalizada)
            time.sleep(0.03)
    return np.stack(conjunto_imagens)


# um dado de entrada rede neural: uma matriz 84x84, onde cada elemento é um pixel e o número é a cor normalizada
stack = captura(monitor)


# -------------------------------****
# Função detecção de game over
# -------------------------------****

def game_over(frame_stack):
    frame = frame_stack[-1]
    region = frame[30:60, 20:160]

    return region.mean() < 0.1


# -------------------------------****
# Função recompensa
# -------------------------------****


def recompensa(done):
    if done:
        return -10.0
    return 0.1


# -------------------------------------------------------------------
# criando rede neural com 3 saídas (nada, pular, abaixar)
# --------------------------------------------------------------------


class Dino(nn.Module):
    def __init__(self):
        super(Dino, self).__init__()

        # extrai caracteristicas da imagem
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Recebe as informações da imagem e toma decisões
        conv = self.saida_conv()

        self.Linear = nn.Sequential(
            nn.Linear(conv, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def saida_conv(self):
        with torch.no_grad():
            x = torch.zeros(1, 4, 84, 84)
            x = self.conv(x)
            return x.view(1, -1).size(1)

    def forward(self, stack):
        x = self.conv(stack)
        x = x.view(x.size(0), -1)
        return self.Linear(x)


modelo = Dino().to(DEVICE)
target = Dino().to(DEVICE)

target.load_state_dict(modelo.state_dict())
target.eval()

# -------------------------------------------------------------------
# otimizador e função loss
# --------------------------------------------------------------------

otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --------------------****
# executando ação
# ---------------------****


def executar_acao(action):
    if action == 1:
        pyautogui.press("space")

# --------------------****
# escolhendo ação
# ---------------------****


def escolher_acao(state):
    if random.random() < EPSILON:
        return random.randint(0, 1)

    with torch.no_grad():
        q = modelo(state)
        return q.argmax(1).item()


# -------------------------------------------------------------------
# Replay Buffer - memória
# --------------------------------------------------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


MEMORIA_CAPACIDADE = 20000
BATCH_SIZE = 32
memoria = deque(maxlen=MEMORIA_CAPACIDADE)


# -------------------------------------------------------------------
# função de treino
# --------------------------------------------------------------------


def trein(monitor):
    global EPSILON

    for epoch in range(1000):

        # Reiniciar o jogo (pressiona espaço para começar após o Game Over)
        pyautogui.press("space")
        time.sleep(1)  # Espera o jogo carregar

        state = captura(monitor)
        state = torch.tensor(
            state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        total_reward = 0

        while True:

            # escolhendo e executando ação
            action = escolher_acao(state)

            for _ in range(FRAME_SKIP):
                executar_acao(action)

            # observando resultado
            next_state = captura(monitor)
            done = game_over(next_state)
            reward = recompensa(done)
            total_reward += reward

            next_state = torch.tensor(
                next_state, dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            # guardando na memória
            memoria.append((state, action, reward, next_state, done))

            # treinando com batch aleatório
            if len(memoria) > BATCH_SIZE:
                batch = random.sample(memoria, BATCH_SIZE)

                # transformando dados em vetor para a rede neural
                b_states = torch.cat([b[0] for b in batch])
                b_actions = torch.tensor([b[1] for b in batch]).to(DEVICE)
                b_rewards = torch.tensor(
                    [b[2] for b in batch], dtype=torch.float32).to(DEVICE)
                b_next_states = torch.cat([b[3] for b in batch])
                b_dones = torch.tensor(
                    [b[4] for b in batch], dtype=torch.float32).to(DEVICE)

                # Q-values atuais
                q_values = modelo(b_states).gather(
                    1, b_actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target(b_next_states).max(1)[0]
                    q_targets = b_rewards + \
                        (GAMMA * max_next_q * (1 - b_dones))

                loss = loss_fn(q_values, q_targets)

                otimizador.zero_grad()
                loss.backward()
                otimizador.step()

            state = next_state

            if done:
                print(
                    f"Episódio {epoch} | Recompensa: {total_reward:.2f}")
                break

        EPSILON = max(0.05, EPSILON*0.9995)
        # Atualiza a target network
        if epoch % 50 == 0:
            target.load_state_dict(modelo.state_dict())

        time.sleep(0.01)


trein(monitor)
