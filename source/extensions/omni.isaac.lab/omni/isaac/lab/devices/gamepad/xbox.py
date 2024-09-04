# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform.rotation import Rotation

import carb
import omni

from ..device_base import DeviceBase
from xbox360controller import Xbox360Controller


class XboxGamepad(DeviceBase):
    """A gamepad controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a gamepad controller for a robotic arm with a gripper.
    It uses the gamepad interface to listen to gamepad events and map them to the robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Stick and Button bindings:
        ============================ ========================= =========================
        Description                  Stick/Button (+ve axis)   Stick/Button (-ve axis)
        ============================ ========================= =========================
        Toggle gripper(open/close)   X Button                  X Button
        Move along x-axis            Left Stick Up             Left Stick Down
        Move along y-axis            Left Stick Left           Left Stick Right
        Move along z-axis            Right Stick Up            Right Stick Down
        Rotate along x-axis          D-Pad Left                D-Pad Right
        Rotate along y-axis          D-Pad Down                D-Pad Up
        Rotate along z-axis          Right Stick Left          Right Stick Right
        ============================ ========================= =========================

    .. seealso::

        The official documentation for the gamepad interface: `Carb Gamepad Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html>`__.

    """

    def __init__(self, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.6, dead_zone: float = 0.01):
        """Initialize the gamepad layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 1.0.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 1.6.
            dead_zone: Magnitude of dead zone for gamepad. An event value from the gamepad less than
                this value will be ignored. Defaults to 0.01.
        """
        self._stick = Xbox360Controller()
        # (x, y, z, roll, pitch, yaw)
        self._delta_pose_raw = np.zeros([6,])

    def __del__(self):
        """Unsubscribe from gamepad events."""
        

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Xbox Gamepad Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tDevice name: Xbox Controller\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): X\n"
        msg += "\tMove arm along x-axis: Left Stick Up/Down\n"
        msg += "\tMove arm along y-axis: Left Stick Left/Right\n"
        msg += "\tMove arm along z-axis: Right Stick Up/Down\n"
        msg += "\tRotate arm along x-axis: D-Pad Right/Left\n"
        msg += "\tRotate arm along y-axis: D-Pad Down/Up\n"
        msg += "\tRotate arm along z-axis: Right Stick Left/Right\n"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._delta_pose_raw.fill(0.0)

    def add_callback(self, key: carb.input.GamepadInput, func: Callable):
        """Add additional functions to bind gamepad.

        A list of available gamepad keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html>`__.

        Args:
            key: The gamepad button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        # self._additional_callbacks[key] = func
        pass

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from gamepad event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        delta_pos = np.zeros([3,])
        delta_rot = np.zeros([3,])

        if self._stick.button_a.is_pressed:
            delta_pos[:] = [-self._stick.axis_l.y, -self._stick.axis_l.x, -self._stick.axis_r.y]
        else:
            delta_rot[:] = [-self._stick.axis_l.y, -self._stick.axis_l.x, -self._stick.axis_r.y]
        
        delta_pos *= 0.2
        delta_rot *= 0.2
        
        # -- convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        # return the command and gripper state
        return np.concatenate([delta_pos, rot_vec]), self._stick.button_x.is_pressed
