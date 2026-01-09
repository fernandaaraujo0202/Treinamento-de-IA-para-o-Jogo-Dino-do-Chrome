# captura de tela
from stable_baselines3 import DQN
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import BaseCallback
import os
from mss import mss
# enviar comandos
import pydirectinput
import cv2
import numpy as np
# extrair o game over
import pytesseract
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ambiente


class Game(Env):

    # configurando ambiente,  ações e observação
    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)

        self.cap = mss()
        self.game_location = {'top': 170,
                              'left': 60, 'width': 500, 'height': 300}
        self.done_location = {'top': 220,
                              'left': 300, 'width': 350, 'height': 70}

    def step(self, action):
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        if action != 2:
            pydirectinput.press(action_map[action])

        # vendo se o jogo acabou
        done, done_cap = self.get_done()
        terminated = done
        truncated = False
        # pegando a próxima observação
        new_observation = self.get_observation()
        # reward - por sobreviver
        reward = 1
        info = {}

        return new_observation, reward, terminated, truncated, info

    # vizualizar o jogo
    def render(self):
        # captura e transforma em array
        frame = np.array(self.cap.grab(self.game_location))
        # fatia
        frame_rgb = frame[:, :, :3]
        # exibe o frame processado
        cv2.imshow('Game', frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
        return frame_rgb

    def close(self):
        cv2.destroyAllWindows()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(1)
        pydirectinput.click(x=150, y=200)
        pydirectinput.press('space')

        return self.get_observation(), {}

    def get_observation(self):

        # capturando tela e mudando canal
        raw = self.cap.grab(self.game_location)
        raw = np.array(raw)
        raw = (raw)[:, :, :3].astype(np.uint8)
        # preto e branco
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))
        # add canal (padrao)
        channel = np.reshape(resized, (1, 83, 100))
        return channel

    def get_done(self):
        done_cap = self.cap.grab(self.done_location)
        done_cap = np.array(done_cap)
        done_cap = (done_cap)[:, :, :3]

        # textos validos para done (over?)
        done_strings = ['GAME', 'GAHE', 'OVER']

        # aplicando OCR
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done, done_cap


env = Game()

done, done_cap = env.get_done()


# ---------------- testes ----------

obs = env.get_observation()
# plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB))
# plt.show()

done, done_cap = env.get_done()
# print(done)
# plt.imshow(done_cap)
# plt.show()
# print(pytesseract.image_to_string(done_cap))

for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
    print(f"total Reward for episode {episode} is {total_reward}")

# -------------------TRAIN---------------

env_checker.check_env(env)


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n.calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

# ------- criando 2 diretorios: trein e logs


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=300, save_path=CHECKPOINT_DIR)


# -------- Build DQN and Train -----------
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR,
            verbose=1, buffer_size=1200000, learning_starts=0)

# kick of training
model.learn(total_timesteps=1000, callback=callback)

# ---Test out Model
model = DQN.load(os.path.join('train_first', 'best_model'))
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
    print(f"total Reward for episode {episode} is {total_reward}")
