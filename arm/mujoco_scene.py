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

# Vivid, saturated colors chosen to pop against the black background --
# purely cosmetic, edit freely.
ARM_COLORS: dict[str, str] = {
    "healthy": "0.1 0.85 1.0 1",       # electric cyan-blue
    "parkinsons": "1.0 0.15 0.25 1",   # vivid red
}
OBJECT_COLOR = "1.0 0.85 0.1 1"  # bright gold, reads clearly against either arm color

_ARM_BODY_TEMPLATE = """
    <body name="{label}_mount" pos="{x} 0 0">
      <geom name="{label}_pedestal" type="cylinder" size="0.06 0.08" pos="0 0 0.08" material="{label}_mat"/>
      <body name="{label}_base" pos="0 0 0.16">
        <joint name="{label}_base_joint" type="hinge" axis="0 0 1" range="-3.15 3.15" damping="2"/>
        <geom name="{label}_base_geom" type="box" size="0.05 0.05 0.03" material="{label}_mat"/>
        <body name="{label}_shoulder" pos="0 0 0.04">
          <joint name="{label}_shoulder_joint" type="hinge" axis="0 -1 0" range="-0.45 -0.1" damping="2"/>
          <geom name="{label}_upper_arm" type="capsule" fromto="0 0 0 0.28 0 0" size="0.025" material="{label}_mat"/>
          <body name="{label}_wrist" pos="0.28 0 0">
            <geom name="{label}_wrist_geom" type="box" size="0.02 0.05 0.02" material="{label}_mat"/>
            <body name="{label}_finger_l" pos="0 0.05 -0.03">
              <joint name="{label}_claw_l_joint" type="slide" axis="0 -1 0" range="0 0.035" damping="1"/>
              <geom name="{label}_finger_l_geom" type="box" size="0.015 0.01 0.03" material="{label}_mat"/>
            </body>
            <body name="{label}_finger_r" pos="0 -0.05 -0.03">
              <joint name="{label}_claw_r_joint" type="slide" axis="0 1 0" range="0 0.035" damping="1"/>
              <geom name="{label}_finger_r_geom" type="box" size="0.015 0.01 0.03" material="{label}_mat"/>
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
      <geom name="{label}_object_geom" type="box" size="0.013 0.013 0.013" material="object_mat" contype="0" conaffinity="0"/>
    </body>
"""

# Per-label material: same rgba as ARM_COLORS plus a bit of self-illumination
# (emission) so the arms read as bright and glowing against the black floor
# even in shadow, rather than just flat-lit color.
_ARM_MATERIAL_TEMPLATE = (
    '    <material name="{label}_mat" rgba="{color}" '
    'emission="0.45" specular="0.6" shininess="0.4"/>'
)

_SCENE_TEMPLATE = """
<mujoco model="pick_and_place_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.02"/>
  <visual>
    <headlight ambient="0.18 0.18 0.18" diffuse="0.7 0.7 0.7"/>
    <rgba fog="0 0 0 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.05 0.05 0.07" rgb2="0.0 0.0 0.0" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.06 0.06 0.07" rgb2="0.01 0.01 0.02" width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="6 6" reflectance="0.25" shininess="0.3"/>
    <material name="object_mat" rgba="{object_color}" emission="0.5" specular="0.6" shininess="0.4"/>
{materials}
  </asset>
  <worldbody>
    <light pos="0 0.2 1.6" dir="0 -0.1 -1" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/>
    <light pos="0.6 -0.6 1.0" dir="-0.4 0.4 -0.7" diffuse="0.35 0.35 0.4"/>
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
    materials = "\n".join(
        _ARM_MATERIAL_TEMPLATE.format(
            label=label, color=ARM_COLORS.get(label, "0.6 0.6 0.6 1")
        )
        for label in labels
    )
    arm_bodies = "\n".join(
        _ARM_BODY_TEMPLATE.format(label=label, x=offset)
        for label, offset in zip(labels, offsets)
    )
    object_bodies = "\n".join(
        _OBJECT_TEMPLATE.format(label=label, ox=offset + 0.15, oy=0.15, oz=0.025)
        for label, offset in zip(labels, offsets)
    )
    return _SCENE_TEMPLATE.format(
        materials=materials,
        object_color=OBJECT_COLOR,
        bodies=arm_bodies + "\n" + object_bodies,
    )


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
