"""
Layer 2 -- desktop MuJoCo simulation.

Consumes ONLY MotorVector trajectories (arm.motor_contract) -- this module
never imports torch, the RL agents, or pick_and_place_env. Swap the model
entirely and this file doesn't change.

Renders one or two labeled 3-DOF arms side by side in MuJoCo's native
desktop viewer (mjpython on macOS), each with a movable "object" prop it
visibly picks up (claw closes -> object attaches to the gripper) and places
(claw opens -> object drops to table height wherever it was released). A
small Parkinsonian resting tremor (~5 Hz sinusoidal jitter on base/shoulder)
is layered onto any track whose label has a nonzero entry in
TREMOR_AMPLITUDE, purely as a rendering effect -- it never touches the
underlying motor vector, the model, or the hardware path. Finished episodes
auto-replay after a short hold so the viewer keeps demonstrating instead of
freezing on the last frame.

Keyboard playback controls:
    space   pause / resume
    r       request a new episode right away (via on_request_episode)
    1 / 2   toggle visibility of the first / second arm
    3       show both arms

macOS note: MuJoCo's native viewer must run on the main thread. Launch
whatever script constructs an ArmSimulator with `mjpython`, not plain
`python` -- mjpython is installed alongside the mujoco pip package in
.venv/bin.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mujoco
import mujoco.viewer

from arm.motor_contract import MotorVector, to_mujoco_angles
from arm.mujoco_scene import build_scene_xml, joint_names

# Playback pacing: each entry in a trajectory is one RL env step: it is
# spread over SUBSTEPS_PER_STEP rendered frames so joint motion (including
# the claw snapping open/closed) looks smooth instead of jumping instantly.
PLAYBACK_HZ = 50.0
SUBSTEPS_PER_STEP = 15

# claw value (0=open .. 1=closed) at/above which the gripper is considered
# "holding" the object, for pick/place visualization purposes.
CLAW_CLOSED_THRESHOLD = 0.5

# World-frame height of the tabletop the object rests on when not held.
# Matches the object's z in mujoco_scene.py's _OBJECT_TEMPLATE default pose.
TABLE_Z = 0.025

# Parkinsonian resting tremor is characteristically ~4-6 Hz. Amplitude is
# kept small so it reads as a shake, not a malfunction -- purely cosmetic,
# edit freely (this was toned down from an earlier, too-strong 0.05 rad).
TREMOR_HZ = 5.0
TREMOR_AMPLITUDE: dict[str, dict[str, float]] = {
    "healthy": {"base": 0.0, "shoulder": 0.0, "claw": 0.0},
    "parkinsons": {"base": 0.02, "shoulder": 0.02, "claw": 0.0},
}
_TREMOR_PHASE_OFFSET = {"base": 0.0, "shoulder": 2.4, "claw": 4.8}

# How long a finished episode holds on its last frame before auto-replaying.
DEFAULT_REPLAY_HOLD_SECONDS = 1.5

_IDLE_MOTORS = MotorVector(base=0.5, shoulder=0.5, claw=0.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_motors(a: MotorVector, b: MotorVector, t: float) -> MotorVector:
    return MotorVector(
        base=_lerp(a.base, b.base, t),
        shoulder=_lerp(a.shoulder, b.shoulder, t),
        claw=_lerp(a.claw, b.claw, t),
    )


@dataclass
class _ArmTrack:
    label: str
    joint_qpos_adr: dict[str, list[int]]
    geom_ids: list[int]
    wrist_body_id: int
    object_mocap_id: int
    trajectory: list[MotorVector] = field(default_factory=lambda: [_IDLE_MOTORS])
    cursor: int = 0
    sub: int = 0
    visible: bool = True
    finished_announced: bool = False
    last_motors: MotorVector = field(default_factory=lambda: _IDLE_MOTORS)


class ArmSimulator:
    """Drives one or two labeled 3-DOF arms from precomputed motor trajectories."""

    def __init__(
        self,
        labels: tuple[str, ...],
        on_request_episode: Optional[Callable[[str, int], list[MotorVector]]] = None,
        enable_tremor: bool = True,
        auto_replay: bool = True,
        replay_hold_seconds: float = DEFAULT_REPLAY_HOLD_SECONDS,
    ):
        if not labels:
            raise ValueError("ArmSimulator requires at least one arm label")
        self.labels = tuple(labels)
        self._on_request_episode = on_request_episode
        self._enable_tremor = enable_tremor
        self._auto_replay = auto_replay
        self._replay_hold_seconds = float(replay_hold_seconds)
        self._replay_pending_since: Optional[float] = None

        self._paused = False
        self._seed = 0
        self._clock = 0.0
        self._viewer = None  # set once run() launches the GUI thread

        self.xml = build_scene_xml(self.labels)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)

        self._tracks: dict[str, _ArmTrack] = {
            label: _ArmTrack(
                label=label,
                joint_qpos_adr=self._resolve_qpos_addrs(label),
                geom_ids=self._resolve_geom_ids(label),
                wrist_body_id=mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"{label}_wrist"
                ),
                object_mocap_id=self.model.body_mocapid[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{label}_object")
                ],
            )
            for label in self.labels
        }
        self._apply_all()

    # -- setup helpers ----------------------------------------------------

    def _resolve_qpos_addrs(self, label: str) -> dict[str, list[int]]:
        addrs: dict[str, list[int]] = {}
        for contract_name, mjnames in joint_names(label).items():
            addrs[contract_name] = [
                int(self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                ])
                for n in mjnames
            ]
        return addrs

    def _resolve_geom_ids(self, label: str) -> list[int]:
        prefix = label + "_"
        ids = []
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is not None and name.startswith(prefix):
                ids.append(i)
        return ids

    # -- public API ---------------------------------------------------------

    def set_trajectory(self, label: str, trajectory: list[MotorVector]) -> None:
        track = self._tracks.get(label)
        if track is None:
            raise KeyError(f"Unknown arm label {label!r}; scene has {self.labels}")
        track.trajectory = list(trajectory) or [_IDLE_MOTORS]
        track.cursor = 0
        track.sub = 0
        track.finished_announced = False
        self._rest_object_at_pick_point(track)

    def _rest_object_at_pick_point(self, track: _ArmTrack) -> None:
        """Move the (not-yet-rendered) object to wherever the arm will first
        close its claw in this trajectory, so the pickup looks like the
        gripper arriving at the object rather than the object teleporting
        into the gripper. If the claw never closes, leave the object as-is.

        The object rests AT TABLE HEIGHT under that XY point, not at the
        gripper's (elevated) Z -- otherwise it visibly floats above the
        table until grabbed.

        This only touches qpos/mocap_pos for bookkeeping; the very next
        _apply_all() call (from the normal render loop, at cursor=0) fully
        overwrites qpos again, so nothing visibly jumps.
        """
        pick_motors = next(
            (m for m in track.trajectory if m.claw >= CLAW_CLOSED_THRESHOLD), None
        )
        if pick_motors is None:
            return
        angles = to_mujoco_angles(pick_motors)
        for contract_name, addrs in track.joint_qpos_adr.items():
            for adr in addrs:
                self.data.qpos[adr] = angles[contract_name]
        mujoco.mj_forward(self.model, self.data)
        wrist_pos = self.data.xpos[track.wrist_body_id]
        self.data.mocap_pos[track.object_mocap_id] = [wrist_pos[0], wrist_pos[1], TABLE_Z]

    def toggle_visible(self, label: str) -> None:
        track = self._tracks.get(label)
        if track is None:
            return
        self._set_visible(track, not track.visible)

    def _set_visible(self, track: _ArmTrack, visible: bool) -> None:
        track.visible = visible
        alpha = 1.0 if visible else 0.0
        for gid in track.geom_ids:
            rgba = self.model.geom_rgba[gid].copy()
            rgba[3] = alpha
            self.model.geom_rgba[gid] = rgba

    def request_new_episode(self) -> None:
        if self._on_request_episode is None:
            print("[sim] no on_request_episode callback wired up; ignoring 'r'")
            return
        self._seed += 1
        for label in self.labels:
            self.set_trajectory(label, self._on_request_episode(label, self._seed))

    def toggle_pause(self) -> None:
        self._paused = not self._paused

    # -- stepping -------------------------------------------------------------

    def _apply_all(self) -> None:
        for track in self._tracks.values():
            self._apply_track(track)
        mujoco.mj_forward(self.model, self.data)
        for track in self._tracks.values():
            self._update_object(track)
        mujoco.mj_forward(self.model, self.data)

    def _apply_track(self, track: _ArmTrack) -> None:
        traj = track.trajectory
        i = min(track.cursor, len(traj) - 1)
        j = min(i + 1, len(traj) - 1)
        t = track.sub / max(SUBSTEPS_PER_STEP, 1)
        motors = _lerp_motors(traj[i], traj[j], t)
        track.last_motors = motors

        angles = to_mujoco_angles(motors)
        if self._enable_tremor:
            amp_cfg = TREMOR_AMPLITUDE.get(track.label)
            if amp_cfg:
                for joint, amp in amp_cfg.items():
                    if amp:
                        phase = (
                            2.0 * math.pi * TREMOR_HZ * self._clock
                            + _TREMOR_PHASE_OFFSET.get(joint, 0.0)
                        )
                        angles[joint] += amp * math.sin(phase)

        for contract_name, addrs in track.joint_qpos_adr.items():
            value = angles[contract_name]
            for adr in addrs:
                self.data.qpos[adr] = value

    def _update_object(self, track: _ArmTrack) -> None:
        """Attach the object to the gripper while held; drop it to table
        height (keeping whatever XY it was released at) once the claw opens,
        so it never floats -- it's either in the gripper or on the table.
        """
        is_closed = track.last_motors.claw >= CLAW_CLOSED_THRESHOLD
        if is_closed:
            self.data.mocap_pos[track.object_mocap_id] = self.data.xpos[track.wrist_body_id]
        else:
            current = self.data.mocap_pos[track.object_mocap_id]
            self.data.mocap_pos[track.object_mocap_id] = [current[0], current[1], TABLE_Z]
        # else: leave mocap_pos exactly where it was last set (or its XML
        # default, if the claw has never closed yet this episode).

    def _advance(self) -> None:
        for track in self._tracks.values():
            if len(track.trajectory) <= 1:
                continue
            track.sub += 1
            if track.sub >= SUBSTEPS_PER_STEP:
                track.sub = 0
                if track.cursor < len(track.trajectory) - 1:
                    track.cursor += 1
                elif not track.finished_announced:
                    track.finished_announced = True
                    print(f"[{track.label}] episode finished after {track.cursor} steps")

    def _all_finished(self) -> bool:
        real_tracks = [t for t in self._tracks.values() if len(t.trajectory) > 1]
        if not real_tracks:
            return False
        return all(t.finished_announced for t in real_tracks)

    def _maybe_auto_replay(self) -> None:
        if not self._auto_replay or self._on_request_episode is None:
            self._replay_pending_since = None
            return
        if self._all_finished():
            if self._replay_pending_since is None:
                self._replay_pending_since = self._clock
            elif self._clock - self._replay_pending_since >= self._replay_hold_seconds:
                self.request_new_episode()
                self._replay_pending_since = None
        else:
            self._replay_pending_since = None

    # -- keyboard controls --------------------------------------------------

    def _key_callback(self, keycode: int) -> None:
        """Entry point MuJoCo calls on ITS OWN GUI thread, separate from the
        thread running run()'s main loop below. Every mjData/mjModel mutation
        must go through self._viewer.lock() to avoid racing that other
        thread -- without it, pressing any key that touches qpos/geom_rgba
        (which is most of them) can corrupt shared state and crash the whole
        process. The try/except is a second line of defense: an exception
        escaping a native callback like this can also take the process down,
        so nothing is allowed to propagate out of here.
        """
        try:
            self._handle_key(keycode)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[sim] key handler error (ignored): {exc!r}")

    def _handle_key(self, keycode: int) -> None:
        if self._viewer is None:
            return
        key = chr(keycode) if 0 <= keycode < 256 else ""
        with self._viewer.lock():
            if key == " ":
                self.toggle_pause()
                print("[sim] " + ("paused" if self._paused else "resumed"))
            elif key.lower() == "r":
                self.request_new_episode()
            elif key in "12" and len(self.labels) > 1:
                idx = int(key) - 1
                if idx < len(self.labels):
                    self.toggle_visible(self.labels[idx])
            elif key == "3":
                for track in self._tracks.values():
                    if not track.visible:
                        self._set_visible(track, True)

    # -- main loop ------------------------------------------------------------

    def run(self) -> None:
        """Launch the desktop viewer and block until the window is closed."""
        print("=" * 60)
        print(f"Arm sim -- showing: {', '.join(self.labels)}")
        print("Controls: [space]=pause/resume  r=new episode now  "
              "1/2=toggle arm visibility  3=show both")
        print(f"Finished episodes auto-replay after {self._replay_hold_seconds}s.")
        print("=" * 60)
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback
        ) as viewer:
            self._viewer = viewer
            period = 1.0 / PLAYBACK_HZ
            try:
                while viewer.is_running():
                    start = time.time()
                    # Everything that can touch qpos/mocap_pos/geom_rgba (or
                    # trigger a new episode, which touches them too) must be
                    # inside this lock -- _key_callback fires on a different
                    # thread and takes the same lock before mutating anything.
                    with viewer.lock():
                        if not self._paused:
                            self._advance()
                            self._maybe_auto_replay()
                        self._clock += period
                        self._apply_all()
                    viewer.sync()
                    elapsed = time.time() - start
                    time.sleep(max(0.0, period - elapsed))
            finally:
                self._viewer = None
