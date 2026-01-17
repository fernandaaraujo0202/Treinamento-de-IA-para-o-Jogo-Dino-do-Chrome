# -----imports----

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import os
import time
import cv2
import numpy as np
import pytesseract
import pydirectinput

from mss import mss
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# --- tesseract ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- o ambiente ---
class Game(Env):
    def __init__(self):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(83, 100, 1), dtype=np.uint8)
        # 0 = pular | 1 = abaixar | 2 = não fazer nada
        self.action_space = Discrete(3)

        self.cap = mss()
        self.game_location = {
            'top': 170,
            'left': 60,
            'width': 500,
            'height': 300
        }
        self.done_location = {
            'top': 220,
            'left': 300,
            'width': 350,
            'height': 70
        }

    # --- step ---
    def step(self, action):
        action_map = {
            0: 'space',
            1: 'down',
            2: None
        }

        if action_map[action] is not None:
            pydirectinput.press(action_map[action])

        done, _ = self.get_done()

        obs = self.get_observation()

        # reward
        reward = 1
        if done:
            reward = -10

        terminated = done
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    # --- RESET ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        time.sleep(1)
        pydirectinput.click(x=150, y=200)
        pydirectinput.press('space')

        obs = self.get_observation()
        info = {}

        return obs, info

    # --- render ---
    def render(self):
        frame = np.array(self.cap.grab(self.game_location))
        frame = frame[:, :, :3]
        cv2.imshow("Game", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()

    # --- observation ---
    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))
        raw = raw[:, :, :3]

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))

        obs = resized.reshape(83, 100, 1).astype(np.uint8)

        return obs

    # --- done ---
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))
        done_cap = done_cap[:, :, :3]

        done_strings = ['GAME', 'GAHE', 'OVER']
        text = pytesseract.image_to_string(done_cap)[:4]

        done = text in done_strings
        return done, done_cap


# --- env setup ---
env = DummyVecEnv([lambda: Game()])
env = VecTransposeImage(env)

check_env(Game(), warn=True)


# --- call back ---
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, f"best_model_{self.n_calls}"
            )
            self.model.save(model_path)
        return True


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"
callback = TrainAndLoggingCallback(300, CHECKPOINT_DIR)


# --- model ---
model = DQN(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    buffer_size=30_000,
    learning_starts=1_000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1_000
)

# --- trein ---
model.learn(total_timesteps=10_000, callback=callback)

# --- test ---

model = DQN.load("./train/best_model_300", env=env)

for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode} | Reward: {total_reward}")
