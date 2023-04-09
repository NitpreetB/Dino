from mss import mss 
import pydirectinput 
import cv2 
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time 
from gym import Env
from gym.spaces import Box, Discrete 
#from ctypes import windll

##create enviro 


class WebGame(Env):
    #setup function
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=225, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # define extraction parameters for the game 
        self.cap = mss()
        self.game_location = {'top':300, 'left':0, 'width':600, 'height':500}
        self.done_location = {'top':405, 'left':630, 'width':660, 'height':90}
    #what is called to do something in the game

    def step(self, action):
        #action key - 0 = Space , 1 = Duck(down) , 2 = No action (no op )
        action_map = {
            0:'space',
            1:'down',
            2:'no_op'
        }
        
        if action != 2:
            pydirectinput.press(action_map[action])
        # checking wheather the game is done
        done, done_cap = self.get_done()
        # Get the next observation 
        new_observation = self.get_observation()
        #reward - get a point for every frame that we are alive
        rewards = 1 
        # info dictionary 
        info = {} 


        return new_observation, rewards, done, info

    #visualize the game 
    def render(self):
        cv2.imshow("Game", np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    #restart the game 
    def reset (self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()
    
    #this closes down the observation 
    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self):
        #get screen capture of game

        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        grey=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grey,(100,83))
        channel=np.reshape(resized,(1,83,100))
        return channel

    #get the done text using OCR
    def get_done(self):
        #get done screen 
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]

        #Valid Done text 
        done_strings = ['GAME', 'GAHE', 'GARE']

        done = False
        res = pytesseract.image_to_string(done_cap)[:4]

        if res in done_strings:
            done = True

        return done, done_cap

    

env = WebGame()
##obs = env.get_observation()
##plt.imshow(cv2.cvtColor((env.get_observation()[0]), cv2.COLOR_BGR2RGB))
#plt.show() 
##done_cap = env.get_done()
##print(done_cap)
##pytesseract.image_to_string(done_cap)[:4]


# test enviroment 
#for episode in range(10):
 #   obs = env.reset()
  #  done= False
   # total_reward = 0 
    
    #while not done:
     #   obs, reward, done , info  = env.step(env.action_space.sample())
      #  total_reward += reward 
    #print (f'Total Reward for epoch {episode} is {total_reward}')  


##Train the Model

#creating a call back 

import os 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker

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
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=300,save_path=CHECKPOINT_DIR)

# build and train DQN

from stable_baselines3 import DQN
model  = DQN ('MlpPolicy', env , tensorboard_log= LOG_DIR, verbose=1, buffer_size=100, learning_starts=0)

#load pre-trained model

model=DQN.load(os.path.join('pretrained_models','best_model_88000'))


#test.py

for episode in range(10):
    obs = env.reset()
    done= False
    total_reward = 0 
     
    while not done:
        action, null = model.predict(obs)
        obs, reward, done , info  = env.step(int(action))
        total_reward += reward 
    print (f'Total Reward for epoch {episode} is {total_reward}')  