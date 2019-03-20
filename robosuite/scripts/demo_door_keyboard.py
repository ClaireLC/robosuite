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
      "JR2Door",
      has_renderer        = True,
      use_camera_obs      = False,
      ignore_done         = True,
      control_freq        = args.control_freq,
      horizon             = args.horizon,
      door_type           = args.door_type,
      arena               = args.arena,
      bot_motion          = "mmp",
      robot_pos           = args.robot_pos,
      dist_to_handle_coef = args.rcoef_dist_to_handle,
      door_angle_coef     = args.rcoef_door_angle,
      handle_con_coef     = args.rcoef_handle_con,
      body_door_con_coef  = args.rcoef_body_door_con,
      self_con_coef       = args.rcoef_self_con,
      arm_handle_con_coef = args.rcoef_arm_handle_con,
      force_coef          = args.rcoef_force,
      gripper_touch_coef  = args.rcoef_gripper_touch,
      debug_print         = args.print_info,
      eef_type            = args.eef_type,
      init_distance       = args.distance,
  )
  
  # initialize device
  if args.device == "keyboard":
      from robosuite.devices import MyKeyboard

      device = MyKeyboard()
      env.viewer.add_keypress_callback("any", device.on_press)
      env.viewer.add_keyup_callback("any", device.on_release)
      env.viewer.add_keyrepeat_callback("any", device.on_press)
  else:
      raise Exception(
          "Invalid device choice: choose 'keyboard'"
      )

  if args.eef_type == "gripper":
    action_vel = np.zeros(9)
  else:
    action_vel = np.zeros(8)
  
  env.reset()
  #init_qpos = np.array([0.39814922,-0.44719879,0.73021735,-1.63751285,1.77351342,2.30105636,4.70869331,-0.99113772,-2.1977303])
  #env.sim.data.qpos[env._ref_joint_pos_indexes] = init_qpos
  env.render()
  
  device.start_control()
  
  while True:
    state = device.get_controller_state()
    for q, qvel in state.items():
      action_vel[q-1] = qvel 
    obs, reward, done, _ = env.step(action_vel)
    #print(env.sim.data.qpos[env._ref_joint_pos_indexes])
    env.render()
    #time.sleep(0.1)
