import argparse
import json
import os
from gym import logger

def custom_arg_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # try to load default config file
  default_path = './default_door.json'
  parser.add_argument("--config_file", help=".json file to load parameters from", default=default_path)
 
  parser.add_argument("--model", help="Directory containing a previously trained model", type=str)
  parser.add_argument("--visualize", help="Visualize the training process", type=str2bool, const=True, nargs='?')
  parser.add_argument("--replay", help="Replay policy of given model, without modifying", type=str2bool, const=True, nargs='?')
  parser.add_argument("--stochastic_replay", help="Whether or not to load a stochastic model", type=str2bool, const=True, nargs='?')
  parser.add_argument("--slurm", help="Whether or not you are training on cluster", type=str2bool, const=True, nargs='?')
  parser.add_argument("--job_id", help="SLURM job id", type=int)

  # Environment parameters
  parser.add_argument("--bot_motion", help="Type of robot motion (static or mobile base)", type=str, choices=['static','mmp'])
  parser.add_argument("--door_type", help="Door type to use", type=str, choices=['dpnl','dpwl'])
  parser.add_argument("--arena", help="Arena type, either empty or room", type=str, choices=['e','r'])
  parser.add_argument("--distance", help="Distance from robot to door", type=str, choices=['touching','near','far'])
  parser.add_argument("--robot_pos", help="Position of door [x,y,z]", nargs='+', type=float)
  parser.add_argument("--door_pos", help="Position of door [x,y,z]", nargs='+', type=float)
  parser.add_argument("--door_quat", help="Position of door [w,x,y,z]", nargs='+', type=float)
  parser.add_argument("--control_freq", help="Control frequency", type=int)

  # Learning parameters
  parser.add_argument("--rl_alg", help="RL algorithm to use", type=str, choices=['ppo1','ppo2'])
  parser.add_argument("--policy", help="Policy model to use", type=str)
  parser.add_argument("--n_steps", help="Number of steps to run for each environment per update", type=int)
  parser.add_argument("--horizon", help="Number of steps before environment resets", type=int)
  parser.add_argument("--opt_epochs", help="Number of epochs when optimizing surrogate", type=int)
  parser.add_argument("--minibatches", help="Number of training minibatches per update", type=int)
  parser.add_argument("--ent_coef", help="Entropy coefficient for loss", type=float)
  parser.add_argument("--vf_coef", help="Value function coefficient for loss", type=float)
  parser.add_argument("--clip_range", help="Clipping parameter", type=float)
  parser.add_argument("--gamma", help="Discount factor", type=float)
  parser.add_argument("--lam", help="Factor for trade-off of bias vs variance for GAE", type=float)
  parser.add_argument("--verbose", help="Verbosity level", type=int)
  parser.add_argument("--total_timesteps", help="Total timesteps to train", type=int)  
  parser.add_argument("--n_cpu", help="Number of CPUs to train on", type=int)  

  return parser

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def serialize_args(args):
  ret = vars(args)["distance"]
  if vars(args)["job_id"] is not None:
    ret += "_" + "{}".format(vars(args)["job_id"])

  for key, value in vars(args).items():
    # skip config file etc. path names
    if type(value) == str:
      if '/' in value: continue
      if 'log_suffix' == key: continue
      if 'log_dir' == key: continue

    if value is None: continue
    
    if key == "total_timesteps": continue
    if key == "visualize": continue
    if key == "verbose": continue
    if key == "distance": continue
    if key == "job_id": continue

    # Necessary to deal with length
    splits = key.split('_')
    short_key = ""

    for split in splits:
      short_key += split[0]
    ret += "_{}.{}".format(short_key, value)
  if ret != "" and ret[-1] == '_':
    ret = ret[:-1]
  ret = ret.replace(" ", "")
  ret = ret.replace(",", ".")
  ret = ret.replace("[", "")
  ret = ret.replace("]", "")
  return ret

def load_defaults(args):
  default_path = './default_door.json'
  default_config_file = default_path
  specified_config_file = args.config_file

  updateable_args_dict = vars(args)
  
  # Load the config file, if there is one specified
  if args.config_file is not None:
    with open(specified_config_file) as f:
      specified_args = json.load(f)
      for key in updateable_args_dict.keys():
        if key in specified_args:
          if updateable_args_dict[key] is None:
            updateable_args_dict[key] = specified_args.get(key)      
        else:
          print("Argument " + key + " not specified in config file")

  # Fill in all the None (unspecified)  args from default config file 
  with open(default_config_file) as def_f:
    default_args = json.load(def_f)
    for key in updateable_args_dict.keys():
      if key in default_args:
        if updateable_args_dict[key] is None:
          updateable_args_dict[key] = default_args.get(key)      
      else:
        print("Argument " + key + " not defined")

