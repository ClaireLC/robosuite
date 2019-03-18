import argparse
import robosuite as suite
import numpy as np
import time
from robosuite.scripts.custom_parser import custom_arg_parser, load_defaults, serialize_args

if __name__ == "__main__":
  
  custom_parser = custom_arg_parser()
  args = custom_parser.parse_args()
  load_defaults(args)
  print(args)
  print(serialize_args(args))

  env = suite.make(
      #"JR2StaticArmDoor",
      "JR2Door",
      has_renderer=True,
      use_camera_obs=False,
      ignore_done=True,
      control_freq=20,
      door_type="dpnl",
      robot_pos=args.robot_pos,
      dist_to_handle_coef = args.rcoef_dist_to_handle,
      door_angle_coef     = args.rcoef_door_angle,
      handle_con_coef     = args.rcoef_handle_con,
      body_door_con_coef  = args.rcoef_body_door_con,
      self_con_coef       = args.rcoef_self_con,
      arm_handle_con_coef = args.rcoef_arm_handle_con,
      arm_door_con_coef   = args.rcoef_arm_door_con,
      force_coef          = args.rcoef_force,
      reset_on_large_force= args.reset_on_large_force,
      debug_print         = args.print_info,
      init_distance       = args.distance,
      eef_type            = args.eef_type,
  )
  
  env.reset()
  env.render()
  
  if args.eef_type == "gripper":
    action_vel = np.zeros(9)
  else:
    action_vel = np.zeros(8)
    
  while True:
    action_vel[0] = 0
    action_vel[1] = 1.0
    if args.eef_type == "gripper":
      action_vel[8] = 0.0
    obs, reward, done, _ = env.step(action_vel)
    env.render()
    #print(reward)
    #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
    #time.sleep(0.05)
