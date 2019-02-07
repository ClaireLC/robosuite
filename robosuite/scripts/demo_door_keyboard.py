import argparse
import robosuite as suite
import numpy as np
import time

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", type=str, default="keyboard")
  args = parser.parse_args()

  env = suite.make(
      "JR2Door",
      has_renderer=True,
      use_camera_obs=False,
      ignore_done=True,
      control_freq=20
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

  env.reset()
  qpos = np.zeros(8)
  qpos[3] = np.pi  
  qpos[4] = np.pi  
  #env.set_robot_joint_positions([qpos])
  env.render()
  #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
  action_vel = [0.0,0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0,0.0]
  device.start_control()
  
  while True:
    state = device.get_controller_state()
    new_vel, joint_num = (state["vel"], state["joint_num"])
    action_vel[joint_num-1] = new_vel
    obs, reward, done, _ = env.step(action_vel)
    env.render()
    #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
    #time.sleep(0.1)
