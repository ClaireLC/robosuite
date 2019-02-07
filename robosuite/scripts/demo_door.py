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
  
  env.reset()
  qpos = np.zeros(8)
  #qpos = np.zeros(10)
  #qpos[2] = np.pi / 2
  qpos[3] = np.pi  
  qpos[4] = np.pi  
  #env.set_robot_joint_positions([qpos])
  env.render()
  print(env.sim.data.qpos[env._ref_joint_vel_indexes])
  
  while True:
    action_vel = [-0.1,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0]
    obs, reward, done, _ = env.step(action_vel)
    env.render()
    print(reward)
    #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
    #time.sleep(0.1)
