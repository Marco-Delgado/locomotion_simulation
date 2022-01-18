"""Test the C++ robot interface.

Follow the
"""

from robot_interface import RobotInterface # pytype: disable=import-error

i = RobotInterface()
print('created robot interface')
o = i.receive_observation()
print('received observation')
