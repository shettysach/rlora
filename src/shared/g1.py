from __future__ import annotations

import numpy as np

LEFT_LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
)
RIGHT_LEG_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_LEG_JOINTS)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINTS)
BODY_JOINTS = (
    LEFT_LEG_JOINTS
    + RIGHT_LEG_JOINTS
    + WAIST_JOINTS
    + LEFT_ARM_JOINTS
    + RIGHT_ARM_JOINTS
)

LEFT_HAND_JOINTS = tuple(
    f"left_hand_{finger}_{index}_joint"
    for finger, count in (("thumb", 3), ("index", 2), ("middle", 2))
    for index in range(count)
)
RIGHT_HAND_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_HAND_JOINTS)
HAND_JOINTS = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

# The Dex3 MJCF declares middle before index. SIMPLE actions use thumb, index, middle.
MJLAB_HAND_FROM_PSI0 = np.array((0, 1, 2, 5, 6, 3, 4, 7, 8, 9, 12, 13, 10, 11))

# MuJoCo/MJLab natural joint order.
DEFAULT_JOINT_POS_MJLAB = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

# For a vector in SONIC/IsaacLab order, select these indices to obtain MuJoCo/MJLab natural order.
MJLAB_FROM_SONIC = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)

# Inverse mapping: MuJoCo/MJLab natural order to SONIC/IsaacLab order.
SONIC_FROM_MJLAB = np.argsort(MJLAB_FROM_SONIC)
