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
  qpos = np.array([4.14576165e-01,-4.83555008e-02, -1.71113779e-01,-1.71592774e+00, 2.74424796e+00, 3.33114205e+00, 1.71887160e+00, 7.30155817e-03, 4.25708286e-05])
  env.set_robot_joint_positions([qpos])
  env.render()
  #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
  action_vel = np.zeros(8)
  device.start_control()
  
  while True:
    state = device.get_controller_state()
    new_vel, joint_num = (state["vel"], state["joint_num"])
    action_vel[joint_num-1] = new_vel
    obs, reward, done, _ = env.step(action_vel)
    env.render()
    #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
    #time.sleep(0.1)
