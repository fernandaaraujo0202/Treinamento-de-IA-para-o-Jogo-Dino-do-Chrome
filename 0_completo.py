import pyautogui
import time
import cv2
import numpy as np
import mss
import torch
import torch.nn as nn
import os
from collections import deque
import random

# ================= CONFIG =================

MODEL_PATH = "dino_dqn.pth"

STACK_SIZE = 4
FRAME_SKIP = 4

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.985

gamma = 0.99
batch_size = 32
lr = 2.5e-4

target_update_freq = 300

# ================= INIT =================

time.sleep(3)
pyautogui.press("space")

frame_stack = deque(maxlen=STACK_SIZE)
frame_anterior = None
frames_parados = 0

# ================= SCREEN =================


def capturar_tela():
    with mss.mss() as sct:
        monitor = {"top": 300, "left": 100, "width": 600, "height": 150}
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (84, 84))
        return img

# ================= ACTION =================


def executar_acao(acao):
    if acao == 1:
        pyautogui.press("space")
    elif acao == 2:
        pyautogui.keyDown("down")
        time.sleep(0.05)
        pyautogui.keyUp("down")

# ================= CACTO =================


def cacto_perto(frame):
    roi = frame[60:120, 40:120]
    return np.mean(roi) < 100

# ================= REWARD =================


def recompensa(game_over, pulou, cacto):
    if game_over:
        return -10
    if cacto and not pulou:
        return -1
    if not cacto and pulou:
        return -1
    return +0.1

# ================= GAME OVER =================


def detectar_game_over(frame):
    """
    Detecta GAME OVER analisando a região central onde o texto aparece.
    """
    # região aproximada onde aparece "GAME OVER"
    roi = frame[30:70, 25:160]

    # média de brilho
    brilho = np.mean(roi)

    # quando o texto aparece, o brilho sobe bastante
    return brilho > 180


# ================= DQN =================


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.net(x)


model = DQN()
target_model = DQN()
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.SmoothL1Loss()

# ================= BUFFER =================


class ReplayBuffer:
    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


buffer = ReplayBuffer()

# ================= ACTION SELECTION =================


def escolher_acao(estado):
    global epsilon
    if random.random() < epsilon:
        return random.randint(0, 2)
    with torch.no_grad():
        estado = torch.from_numpy(estado).unsqueeze(0).float() / 255.0
        q = model(estado)
        return torch.argmax(q).item()

# ================= TRAIN =================


step_count = 0


def treinar_dqn():
    global step_count

    if len(buffer.buffer) < batch_size:
        return

    estados, acoes, recompensas, prox_estados, dones = zip(
        *buffer.sample(batch_size)
    )

    estados = torch.tensor(estados).float() / 255.0
    prox_estados = torch.tensor(prox_estados).float() / 255.0
    acoes = torch.tensor(acoes).long()
    recompensas = torch.tensor(recompensas).float()
    dones = torch.tensor(dones).float()

    q_atual = model(estados).gather(1, acoes.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        q_next = target_model(prox_estados).max(1)[0]

    target = recompensas + gamma * q_next * (1 - dones)

    loss = loss_fn(q_atual, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    step_count += 1
    if step_count % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

# ================= SAVE / LOAD =================


def salvar_modelo():
    torch.save(model.state_dict(), MODEL_PATH)


if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    target_model.load_state_dict(model.state_dict())
    print("Modelo carregado.")

# ================= LOOP FINAL =================

for episode in range(1000):

    time.sleep(0.5)
    pyautogui.press("space")

    frame_anterior = None
    frames_parados = 0

    frame = capturar_tela()
    frame_stack.clear()

    for _ in range(STACK_SIZE):
        frame_stack.append(frame)

    estado = np.stack(frame_stack, axis=0)
    passos = 0

    while True:

        # 1️⃣ Decide ação a partir do estado ATUAL
        if random.random() < 0.7 and cacto_perto(frame):
            acao = 1
        else:
            acao = escolher_acao(estado)

        # 2️⃣ Executa ação (frame skip)
        for _ in range(FRAME_SKIP):
            executar_acao(acao)

        # 3️⃣ Captura novo frame
        frame = capturar_tela()
        frame_stack.append(frame)
        prox_estado = np.stack(frame_stack, axis=0)

        # 4️⃣ Detecta eventos no NOVO frame
        done = detectar_game_over(frame)
        cacto = cacto_perto(frame)

        # 5️⃣ Calcula recompensa
        pulou = (acao == 1)
        r = recompensa(done, pulou, cacto)

        # 6️⃣ Guarda transição
        buffer.add(estado, acao, r, prox_estado, done)

        # 7️⃣ Treina
        if len(buffer.buffer) > 1000:
            treinar_dqn()

        # 8️⃣ Avança estado
        estado = prox_estado
        passos += 1

        # 9️⃣ Finaliza episódio
        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 50 == 0:
        salvar_modelo()
        print(f"Modelo salvo | Episódio {episode}")

    print(
        f"Episódio {episode} | "
        f"Passos: {passos} | "
        f"Epsilon: {epsilon:.3f} | "
        f"Buffer: {len(buffer.buffer)}"
    )
