import robosuite as suite
import numpy as np

if __name__ == "__main__":

  env = suite.make(
      "BaxterDoor",
      has_renderer=True,
  )
  
  env.reset()
  env.viewer.set_camera(camera_id=0)
  
  # do visualization
  for i in range(5000):
    #action = np.random.randn(env.dof)
    #obs, reward, done, _ = env.step(action)
    env.render()
