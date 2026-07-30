"""
MJCF scene builder for the 3-DOF arm visualization.

Builds one or two arm instances (for side-by-side healthy vs. Parkinson's
playback) sharing the same world. Pure model construction -- no viewer, no
model/policy coupling. Each arm exposes exactly 3 controllable joints per
instance, matching the [base, shoulder, claw] motor-vector contract:

    {label}_base_joint      -- hinge, base rotation
    {label}_shoulder_joint  -- hinge, shoulder pitch (axis flipped so a
                               POSITIVE angle lifts the arm UP; the pivot
                               sits 0.20m off the floor and the shoulder's
                               MUJOCO_ANGLE_RANGE is capped so the fingertip
                               can never swing down through the floor)
    {label}_claw_l_joint    -- slide, left finger  (0=open/wide, max=closed/narrow)
    {label}_claw_r_joint    -- slide, right finger (mirrored, same `claw` value)

Each arm also gets a `{label}_object` mocap body -- a movable prop that
arm.sim attaches to the gripper while the claw is closed and drops in place
when it opens, so the viewer actually shows something being picked up and
placed rather than an arm waving at empty air.
"""

from __future__ import annotations

ARM_COLORS: dict[str, str] = {
    "healthy": "0.2 0.55 0.95 1",
    "parkinsons": "0.85 0.25 0.2 1",
}

_ARM_BODY_TEMPLATE = """
    <body name="{label}_mount" pos="{x} 0 0">
      <geom name="{label}_pedestal" type="cylinder" size="0.06 0.08" pos="0 0 0.08" rgba="{color}"/>
      <body name="{label}_base" pos="0 0 0.16">
        <joint name="{label}_base_joint" type="hinge" axis="0 0 1" range="-3.15 3.15" damping="2"/>
        <geom name="{label}_base_geom" type="box" size="0.05 0.05 0.03" rgba="{color}"/>
        <body name="{label}_shoulder" pos="0 0 0.04">
          <joint name="{label}_shoulder_joint" type="hinge" axis="0 -1 0" range="-0.45 -0.1" damping="2"/>
          <geom name="{label}_upper_arm" type="capsule" fromto="0 0 0 0.28 0 0" size="0.025" rgba="{color}"/>
          <body name="{label}_wrist" pos="0.28 0 0">
            <geom name="{label}_wrist_geom" type="box" size="0.02 0.05 0.02" rgba="{color}"/>
            <body name="{label}_finger_l" pos="0 0.05 -0.03">
              <joint name="{label}_claw_l_joint" type="slide" axis="0 -1 0" range="0 0.035" damping="1"/>
              <geom name="{label}_finger_l_geom" type="box" size="0.015 0.01 0.03" rgba="{color}"/>
            </body>
            <body name="{label}_finger_r" pos="0 -0.05 -0.03">
              <joint name="{label}_claw_r_joint" type="slide" axis="0 1 0" range="0 0.035" damping="1"/>
              <geom name="{label}_finger_r_geom" type="box" size="0.015 0.01 0.03" rgba="{color}"/>
            </body>
          </body>
        </body>
      </body>
    </body>
"""

# A free-floating "object" the arm can visibly pick up and place. It's a
# mocap body -- no joints, world-framed -- so arm.sim can reposition it
# directly via data.mocap_pos each frame (attach to the gripper while held,
# freeze in place once released) without it needing to be part of the
# motor-vector contract at all.
_OBJECT_TEMPLATE = """
    <body name="{label}_object" mocap="true" pos="{ox} {oy} {oz}">
      <geom name="{label}_object_geom" type="box" size="0.013 0.013 0.013" rgba="0.95 0.75 0.15 1" contype="0" conaffinity="0"/>
    </body>
"""

_SCENE_TEMPLATE = """
<mujoco model="pick_and_place_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.02"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.95 0.98" rgb2="0.7 0.75 0.85" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.78 0.78 0.78" width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="6 6" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="1.2 0.8 0.02" material="grid_mat"/>
{bodies}
  </worldbody>
</mujoco>
"""


def build_scene_xml(labels: tuple[str, ...], x_spacing: float = 0.4) -> str:
    """Build an MJCF XML string containing one arm body per label.

    labels: e.g. ("healthy",), ("parkinsons",), or ("healthy", "parkinsons").
    """
    n = len(labels)
    offsets = [(i - (n - 1) / 2.0) * x_spacing for i in range(n)]
    arm_bodies = "\n".join(
        _ARM_BODY_TEMPLATE.format(
            label=label,
            x=offset,
            color=ARM_COLORS.get(label, "0.5 0.5 0.5 1"),
        )
        for label, offset in zip(labels, offsets)
    )
    object_bodies = "\n".join(
        _OBJECT_TEMPLATE.format(label=label, ox=offset + 0.15, oy=0.15, oz=0.025)
        for label, offset in zip(labels, offsets)
    )
    return _SCENE_TEMPLATE.format(bodies=arm_bodies + "\n" + object_bodies)


def joint_names(label: str) -> dict[str, list[str]]:
    """Map contract joint names -> MJCF joint names for one arm instance.

    `claw` expands to two mirrored finger joints in the sim; the contract
    itself still only ever carries a single `claw` scalar.
    """
    return {
        "base": [f"{label}_base_joint"],
        "shoulder": [f"{label}_shoulder_joint"],
        "claw": [f"{label}_claw_l_joint", f"{label}_claw_r_joint"],
    }
