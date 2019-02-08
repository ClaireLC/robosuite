import argparse
import robosuite as suite
import numpy as np
import time

from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy,MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO1
from stable_baselines import PPO2

from robosuite.wrappers import GymWrapper

if __name__ == "__main__":
  

  model_name = "ppo_test"
  model_save_path = "../learned_models/" + model_name + "/"
  tb_save_path = "../tb_logs/" +  model_name + "/"
  final_model_path = model_save_path + "final_" + model_name
  show_render = True

  env = GymWrapper(
      suite.make(
      "JR2Door",
      has_renderer=show_render,
      use_camera_obs=False,
      ignore_done=False,
      control_freq=20
    )
  )
  
  print(env.__dict__)
  env = DummyVecEnv([lambda: env])
  #env = SubprocVecEnv([lambda: env for i in range(1)])

  model = PPO1(MlpPolicy,env, verbose=1,timesteps_per_actorbatch=2000, optim_epochs=10,tensorboard_log=tb_save_path)
  #model = PPO2(MlpPolicy, env, verbose=1,n_steps=2048, nminibatches=32, noptepochs=10,tensorboard_log=tb_save_path)

  model.learn(total_timesteps = 8000,save_dir = model_save_path, render=show_render)
  #model.save(final_model_path)
  #env.render()
  #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
  #model = PPO1.load("models/8192model.ckpt")
  
  print("Done training")
  obs = env.reset()
  quit()
  while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    #print(obs)
    #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
    #time.sleep(0.1)
