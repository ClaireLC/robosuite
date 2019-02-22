import argparse
import sys
import numpy as np
import time
import json

import robosuite as suite
from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy,MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO1
from stable_baselines import PPO2

from robosuite.wrappers import GymWrapper
from robosuite.scripts.custom_parser import custom_arg_parser, load_defaults, serialize_args

def main():

  parser = custom_arg_parser()
  args = parser.parse_args()
  load_defaults(args)
  print(args)
  print(serialize_args(args))
  # Create the model name with all the parameters
  
  model_name = serialize_args(args)
  model_save_path = "../../learned_models/" + model_name + "/"
  tb_save_path = "../../tb_logs/" +  model_name + "/"
  final_model_path = model_save_path + "final-" + model_name
  show_render = args.visualize

  env = GymWrapper(
      suite.make(
      "JR2Door",
      has_renderer=show_render,
      use_camera_obs=False,
      ignore_done=False,
      control_freq=args.control_freq,
      horizon=args.horizon,
      door_type=args.door_type,
      arena=args.arena,
      bot_motion=args.bot_motion,
      robot_pos=args.robot_pos
    )
  )
  
  if args.slurm:
    env = SubprocVecEnv([lambda: env for i in range(args.n_cpu)])
  else:
    env = DummyVecEnv([lambda: env])

  # Load the specified model, if there is one
  if args.model is not None:
    # Training from checkpoint, so need to reset timesteps for tb
    reset_num_timesteps = False
    if args.rl_alg == "ppo2":
      model = PPO2.load(args.model,env=env)
    if args.rl_alg == "ppo1":
      model = PPO1.load(args.model,env=env)
  else: 
    # New model, so need to reset timesteps for tb
    reset_num_timesteps = True
    if args.rl_alg == "ppo2":
      model = PPO2(
                  args.policy,
                  env,
                  verbose=args.verbose,
                  n_steps=args.n_steps,
                  nminibatches=args.minibatches,
                  noptepochs=args.opt_epochs,
                  cliprange=args.clip_range,
                  ent_coef=args.ent_coef,
                  tensorboard_log=tb_save_path
                  )

    elif args.rl_alg == "ppo1":
      model = PPO1(
                  args.policy,
                  env,
                  verbose=args.verbose,
                  timesteps_per_actorbatch=args.n_steps,
                  optim_epochs=args.opt_epochs,
                  tensorboard_log=tb_save_path,
                  )
  if args.replay:
    quit()
    # Replay a policy
  else:
    # Train
    model.learn(
                total_timesteps = args.total_timesteps,
                save_dir = model_save_path,
                render=show_render,
                reset_num_timesteps=reset_num_timesteps,
                )

    model.save(final_model_path)
  
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

if __name__ == "__main__":
  main()

