"""
Driver class for Keyboard controller.
"""

import glfw
import numpy as np
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class MyKeyboard(Device):
    """A Keyboard driver class to command JR2 with joint velocities."""

    def __init__(self):
        """
        Initialize a Keyboard device.
        """

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._vel_step = 0.1
    
        self.state = {
                      1: 0.0,
                      2: 0.0,  
                      3: 0.0,  
                      4: 0.0,  
                      5: 0.0,  
                      6: 0.0,  
                      7: 0.0,  
                      8: 0.0,  
                      #9: 0.0,  
                      }
  
    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        #print_command("spacebar", "toggle gripper (open/close)")
        print_command("w-a-s-d", "move arm horizontally in x-y plane")
        print_command("r", "set velocity to 0")
        print_command("1-8", "joint to command")
        #print_command("w-a-s-d", "move arm horizontally in x-y plane")
        #print_command("r-f", "move arm vertically")
        #print_command("z-x", "rotate arm about x-axis")
        #print_command("t-g", "rotate arm about y-axis")
        #print_command("c-v", "rotate arm about z-axis")
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.state = {
                      1: 0.0,
                      2: 0.0,  
                      3: 0.0,  
                      4: 0.0,  
                      5: 0.0,  
                      6: 0.0,  
                      7: 0.0,  
                      8: 0.0,  
                      #9: 0.0,  
                      }

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """Returns the current state of the keyboard, a dictionary of pos, orn, grasp, and reset."""
        return self.state

    def on_press(self, window, key, scancode, action, mods):
        """
        Key handler for key presses.
        """
        if key == glfw.KEY_W:
          self.state[self.joint] += self._vel_step
        elif key == glfw.KEY_S:
          self.state[self.joint] -= self._vel_step
        elif key == glfw.KEY_R:
          self.state[self.joint]  = 0.0
  
        if key == glfw.KEY_1:
          self.joint  = 1
        elif key == glfw.KEY_2:
          self.joint  = 2
        elif key == glfw.KEY_3:
          self.joint  = 3
        elif key == glfw.KEY_4:
          self.joint  = 4
        elif key == glfw.KEY_5:
          self.joint  = 5 
        elif key == glfw.KEY_6:
          self.joint  = 6
        elif key == glfw.KEY_7:
          self.joint  = 7
        elif key == glfw.KEY_8:
          self.joint  = 8
        elif key == glfw.KEY_9:
          self.joint  = 9

    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.
        """

        # controls for grasping
        if key == glfw.KEY_SPACE:
            self.grasp = not self.grasp  # toggle gripper

        # user-commanded reset
        elif key == glfw.KEY_Q:
            self._reset_state = 1
            self._enabled = False
            self._reset_internal_state()
