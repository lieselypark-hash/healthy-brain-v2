"""Critic value + actor policy heatmaps for the pick-and-place A2C agent.

Local, self-contained version of `heatmap.ipynb` — no Colab, no uploads. Clone the
repository, point `--checkpoint` at any `.pt` file, and run:

    python heatmap.py
    python heatmap.py --checkpoint "checkpoints/a2c_rpe_final (2).pt"
    python heatmap.py --object 4 0 --target 0 4 --grid-size 5

Output is deliberately quiet: only the figures are shown (critic value maps, the
holding difference map, and the 8-panel actor policy grid). Pass `--stats` for the
numeric diagnostics the notebook printed, or `--no-show` to save PNGs only.

Conventions (unchanged from the notebook):

* Positions are (row, col), matching `PickAndPlaceEnv.agent_pos[0] == row`.
* Observations are built by `PickAndPlaceEnv._get_obs` itself, so normalisation can
  never drift from the training-time encoding.
* With holding=True the environment encodes the object at the agent's own cell (it is
  being carried), so OBJECT_RC does not enter those observations at all. That is the
  only holding=1 encoding the critic ever saw in training.
* Do not compare the two critic panels' overall levels as "which goal is worth more":
  a not-holding state still has +REWARD_PICK ahead of it as well as +REWARD_PLACE, so
  holding=False sitting uniformly higher is expected arithmetic. The within-panel
  gradient carries the spatial information.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Run from anywhere: resolve imports and default paths against the repo root, which is
# this file's directory.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from a2c_rpe_model import A2CAgent  # noqa: E402
from pick_and_place_env import PickAndPlaceEnv  # noqa: E402


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Critic value and actor policy heatmaps for a trained A2C checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", "-c", default=os.path.join("checkpoints", "a2c_rpe_final.pt"),
        help="Path to the .pt checkpoint. Relative paths resolve against the repo root.",
    )
    p.add_argument("--grid-size", type=int, default=5, help="Grid edge length.")
    p.add_argument(
        "--object", nargs=2, type=int, default=(0, 4), metavar=("ROW", "COL"),
        help="Object location as row col (ignored; fixed to 0 4).",
    )
    p.add_argument(
        "--target", nargs=2, type=int, default=(4, 0), metavar=("ROW", "COL"),
        help="Target location as row col (ignored; fixed to 4 0).",
    )
    p.add_argument("--hidden-dim", type=int, default=128, help="Network hidden width.")
    p.add_argument(
        "--outdir", default="results",
        help="Directory for saved PNGs. Relative paths resolve against the repo root.",
    )
    p.add_argument(
        "--no-save", action="store_true", help="Do not write PNGs to --outdir.",
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Do not open figure windows (useful when only the PNGs are wanted).",
    )
    p.add_argument(
        "--independent-scale", action="store_true",
        help="Give each critic panel its own colour scale instead of a shared one.",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print the numeric diagnostics (ranges, sub-goal correlations, greedy-move test).",
    )
    return p.parse_args(argv)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_DIR, path)


def resolve_checkpoint(requested: str) -> str:
    """Return an existing checkpoint path, or fail with the available alternatives."""
    path = _abs(requested)
    if os.path.exists(path):
        return path

    ckpt_dir = os.path.join(REPO_DIR, "checkpoints")
    available = (
        sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        if os.path.isdir(ckpt_dir) else []
    )
    raise FileNotFoundError(
        f"No checkpoint at {requested!r} (looked in {path}).\n"
        f"Available in checkpoints/: {available}\n"
        "Pass one with --checkpoint; quote names containing spaces."
    )


# --------------------------------------------------------------------------------------
# Observations and sweeps
# --------------------------------------------------------------------------------------

def make_observation_builder(env, object_rc, target_rc):
    """Return a fn giving the observation for a synthetic mid-episode state.

    `cue_active` and `task_started` are both forced true: without them the critic would
    be scored on a pre-start state rather than the navigation phase these maps are about.
    """
    def build_observation(agent_rc: tuple[int, int], holding: bool) -> np.ndarray:
        env.agent_pos = np.array(agent_rc, dtype=np.int32)
        env.object_pos = np.array(object_rc, dtype=np.int32)
        env.target_pos = np.array(target_rc, dtype=np.int32)
        env.holding = bool(holding)
        env.object_placed = False
        env.task_started = True
        env._step_count = env.cue_step        # makes cue_active == 1.0
        return env._get_obs()

    return build_observation


def sweep(agent, build_observation, grid_size, device, holding: bool):
    """Return (V, pi) over every agent position: (G, G) values and (G, G, A) probabilities."""
    coords = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    obs = np.stack([build_observation(rc, holding) for rc in coords])

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        action_probs, values = agent.network(obs_t)

    probs = action_probs.cpu().numpy()

    # The actor head already applies softmax. If a future head drops it, this catches the
    # regression instead of silently rendering logits as probabilities.
    sums = probs.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise AssertionError(
            "Actor outputs do not sum to 1 (range "
            f"{sums.min():.4f}..{sums.max():.4f}) - the head may no longer be softmaxed. "
            "Do NOT add a second softmax; fix the source instead."
        )

    V = values.squeeze(-1).cpu().numpy().reshape(grid_size, grid_size)
    P = probs.reshape(grid_size, grid_size, probs.shape[-1])
    return V, P


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------

CMAP = plt.get_cmap("Blues")
MARK_OFFSET = -0.29   # markers sit in the cell corner so they never occlude the value


def _annotate(ax, V, norm, cmap, grid_size) -> None:
    """Print each cell's value, switching ink to stay legible on dark fills."""
    for r in range(grid_size):
        for c in range(grid_size):
            rgba = cmap(norm(V[r, c]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                c, r, f"{V[r, c]:.2f}",
                ha="center", va="center", fontsize=8.5,
                color="#f7f7f7" if luminance < 0.55 else "#1a1a19",
            )


def _mark_sites(ax, object_rc, target_rc, carried: bool) -> None:
    """Overlay the target (star) and object (circle) positions."""
    ax.scatter(
        target_rc[1] + MARK_OFFSET, target_rc[0] + MARK_OFFSET, marker="*", s=230,
        facecolor="#ffffff", edgecolor="#1a1a19", linewidth=1.3,
        zorder=3, label="target",
    )
    if carried:
        # The object rides with the agent in this variant; the marker is a reference
        # position only, so it is drawn hollow to say "not the encoded location".
        ax.scatter(
            object_rc[1] + MARK_OFFSET, object_rc[0] + MARK_OFFSET, marker="o", s=110,
            facecolor="none", edgecolor="#1a1a19", linewidth=1.3, linestyle="--",
            zorder=3, label="object (carried — pickup site)",
        )
    else:
        ax.scatter(
            object_rc[1] + MARK_OFFSET, object_rc[0] + MARK_OFFSET, marker="o", s=110,
            facecolor="#ffffff", edgecolor="#1a1a19", linewidth=1.3,
            zorder=3, label="object",
        )


def _style_axes(ax, title: str, grid_size: int) -> None:
    ax.set_title(title, fontsize=11, pad=10, color="#1a1a19")
    ax.set_xlabel("col", fontsize=9.5, color="#5c5c58")
    ax.set_ylabel("row", fontsize=9.5, color="#5c5c58")
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.tick_params(labelsize=9, colors="#5c5c58", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Recessive 2px surface gap between cells, drawn as minor gridlines.
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=2)
    ax.tick_params(which="minor", length=0)


def plot_value_maps(V_free, V_hold, cfg):
    """Two-panel critic value map, one panel per holding state."""
    g, extent = cfg["grid_size"], cfg["extent"]
    shared = cfg["shared_scale"]
    vmin = float(min(V_free.min(), V_hold.min()))
    vmax = float(max(V_free.max(), V_hold.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, V, holding in ((axes[0], V_free, False), (axes[1], V_hold, True)):
        lo = vmin if shared else float(V.min())
        hi = vmax if shared else float(V.max())
        im = ax.imshow(V, cmap=CMAP, vmin=lo, vmax=hi, origin="upper", extent=extent)
        _annotate(ax, V, im.norm, CMAP, g)
        _mark_sites(ax, cfg["object_rc"], cfg["target_rc"], carried=holding)
        _style_axes(ax, f"V(s)  ·  holding = {holding}", g)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("state value  V(s)", fontsize=9, color="#5c5c58")
        cbar.ax.tick_params(labelsize=8, colors="#5c5c58", length=0)
        cbar.outline.set_visible(False)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
            frameon=False, fontsize=8.5, labelcolor="#5c5c58",
            handletextpad=0.9, columnspacing=1.8,
        )

    scale_note = "shared colour scale" if shared else "independent colour scale per panel"
    fig.suptitle(
        f"Critic state value across the {g}x{g} grid\n"
        f"object {cfg['object_rc']} · target {cfg['target_rc']} · "
        f"{cfg['checkpoint_name']} · {scale_note}",
        fontsize=12.5, y=1.02, color="#1a1a19",
    )
    fig.tight_layout()
    return fig


def plot_difference_map(V_free, V_hold, cfg):
    """V(holding=True) − V(holding=False): what carrying the object changes.

    A diverging ramp only earns its place if the data actually straddles zero. When the
    sign is uniform there is no polarity to encode, so magnitude gets a single-hue ramp
    instead and the constant sign is stated in the title.
    """
    g, extent = cfg["grid_size"], cfg["extent"]
    diff = V_hold - V_free

    if bool(diff.min() < 0 < diff.max()):
        bound = float(np.abs(diff).max()) or 1e-6
        dmap, dlo, dhi = plt.get_cmap("RdBu_r"), -bound, bound
        dtitle = "V(holding=True) − V(holding=False)"
    else:
        sign_word = "lower" if diff.max() <= 0 else "higher"
        dmap = plt.get_cmap("Purples_r" if diff.max() <= 0 else "Purples")
        dlo, dhi = float(diff.min()), float(diff.max())
        dtitle = (
            "V(holding=True) − V(holding=False)\n"
            f"uniformly {sign_word} — no sign change, so a single-hue ramp"
        )

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(diff, cmap=dmap, vmin=dlo, vmax=dhi, origin="upper", extent=extent)
    _annotate(ax, diff, im.norm, dmap, g)
    _mark_sites(ax, cfg["object_rc"], cfg["target_rc"], carried=False)
    _style_axes(ax, dtitle, g)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ value", fontsize=9, color="#5c5c58")
    cbar.ax.tick_params(labelsize=8, colors="#5c5c58", length=0)
    cbar.outline.set_visible(False)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
        frameon=False, fontsize=8.5, labelcolor="#5c5c58",
        handletextpad=0.9, columnspacing=1.8,
    )
    fig.tight_layout()
    return fig, diff


def plot_policy_maps(P_free, P_hold, cfg):
    """Eight panels: four movement actions x two holding states, coloured by pi(a|s).

    These are raw softmax probabilities over all actions, so the four panels in a row do
    not sum to 1 — PICK, PLACE and START hold the remaining mass, most visibly at the
    object cell where every direction correctly goes dim. UP is visually up: it
    decrements the row and origin="upper" puts row 0 at the top.
    """
    g, extent = cfg["grid_size"], cfg["extent"]
    # Fixed 0-1 scale: probabilities are absolute, so the scale stays meaningful across
    # checkpoints and all eight panels are directly comparable.
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 8.0))

    for row, (P, holding) in enumerate(((P_free, False), (P_hold, True))):
        for col, (dname, arrow, aidx) in enumerate(cfg["directions"]):
            ax = axes[row, col]
            panel = P[:, :, aidx]
            im = ax.imshow(
                panel, cmap=CMAP, vmin=0.0, vmax=1.0, origin="upper", extent=extent
            )
            _annotate(ax, panel, im.norm, CMAP, g)
            _mark_sites(ax, cfg["object_rc"], cfg["target_rc"], carried=holding)
            _style_axes(ax, f"{arrow}  {dname}", g)

            # Axis labels only on the outer edge; 8 copies would be noise.
            if row == 0:
                ax.set_xlabel("")
            if col > 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(f"holding = {holding}\n\nrow", fontsize=9.5, color="#1a1a19")

    # One shared colorbar - with a fixed scale, eight identical bars carry no information.
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, aspect=40)
    cbar.set_label("π(a|s)  ·  probability of this action", fontsize=9, color="#5c5c58")
    cbar.ax.tick_params(labelsize=8, colors="#5c5c58", length=0)
    cbar.outline.set_visible(False)

    # One legend for the whole figure rather than per-panel.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        fontsize=9, labelcolor="#5c5c58", handletextpad=0.9, columnspacing=2.0,
        bbox_to_anchor=(0.45, -0.02),
    )
    fig.suptitle(
        f"Actor policy — directional action probabilities ({g}x{g} grid)\n"
        f"object {cfg['object_rc']} · target {cfg['target_rc']} · "
        f"{cfg['checkpoint_name']} · raw π(a|s) over {cfg['action_dim']} actions · "
        f"fixed 0-1 scale",
        fontsize=13, y=0.99, color="#1a1a19",
    )
    return fig


# --------------------------------------------------------------------------------------
# Optional diagnostics (--stats)
# --------------------------------------------------------------------------------------

def print_stats(env, V_free, V_hold, P_free, P_hold, diff, cfg) -> None:
    g = cfg["grid_size"]
    object_rc, target_rc = cfg["object_rc"], cfg["target_rc"]

    def _argmax_rc(V):
        r, c = np.unravel_index(int(np.argmax(V)), V.shape)
        return int(r), int(c)

    header = f"{'variant':<14}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}{'argmax':>10}"
    print(header)
    print("-" * len(header))
    for name, V in (("holding=False", V_free), ("holding=True", V_hold)):
        print(
            f"{name:<14}{V.min():>9.3f}{V.max():>9.3f}{V.mean():>9.3f}{V.std():>9.3f}"
            f"{str(_argmax_rc(V)):>10}"
        )

    # Where the peak lands is a weak diagnostic: with a nearly position-independent critic
    # the argmax is decided by tiny gradients and can coincide with a site by luck. The
    # informative test is whether value actually tracks distance to the current sub-goal.
    rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    d_to_object = np.abs(rr - object_rc[0]) + np.abs(cc - object_rc[1])
    d_to_target = np.abs(rr - target_rc[0]) + np.abs(cc - target_rc[1])

    print("\nDoes value track distance to the CURRENT sub-goal?")
    print("  (negative = closer to the sub-goal is worth more, which is what we want)\n")
    for name, V, d, goal in (
        ("holding=False", V_free, d_to_object, "object"),
        ("holding=True", V_hold, d_to_target, "target"),
    ):
        r = float(np.corrcoef(V.ravel(), d.ravel())[0, 1])
        if r < -0.5:
            verdict = "tracks it"
        elif r < -0.2:
            verdict = "weakly tracks it"
        elif r <= 0.2:
            verdict = "essentially ignores it"
        else:
            verdict = "INVERTED - farther from the sub-goal scores higher"
        print(f"  {name:<14} corr(V, distance to {goal:<6}) = {r:+.3f}   -> {verdict}")

    # How much of the total variation is just the holding bit, versus position?
    gap = abs(float(V_hold.mean() - V_free.mean()))
    within = max(float(V_free.std()), float(V_hold.std()))
    print(f"\n  between-variant gap (holding bit) : {gap:.3f}")
    print(f"  within-variant spread (position)  : {within:.3f}")
    print(f"  ratio                             : {gap / (within + 1e-12):.1f}x")
    if gap > 3 * within:
        print(
            "  -> The holding bit dominates: this critic is close to a two-level step\n"
            "     function, with position contributing little."
        )

    print(
        "\nNote on the direction of the gap. V(holding=False) > V(holding=True) is CORRECT,\n"
        f"not a bug: from a not-holding state the +{env.reward_pick:g} pick reward is still ahead\n"
        f"of you as well as the +{env.reward_place:g} place reward, whereas a holding state has\n"
        "already banked the pick. So not-holding legitimately has more remaining return.\n"
        "It does NOT mean the critic prefers the target over the object."
    )

    print(f"\nΔ range: {diff.min():+.3f} .. {diff.max():+.3f}   mean {diff.mean():+.3f}")

    dir_idx = [idx for _, _, idx in cfg["directions"]]
    print()
    for name, P in (("holding=False", P_free), ("holding=True", P_hold)):
        mass = P[:, :, dir_idx].sum(axis=-1)
        other = 1.0 - mass
        print(
            f"{name:<14} directional mass {mass.min():.3f}..{mass.max():.3f}   "
            f"| pick/place/start holds {other.min():.3f}..{other.max():.3f}"
        )

    header = f"{'variant':<14}{'action':<8}{'mean':>8}{'max':>8}{'argmax':>10}"
    print("\n" + header)
    print("-" * len(header))
    for vname, P in (("holding=False", P_free), ("holding=True", P_hold)):
        for dname, _arrow, aidx in cfg["directions"]:
            panel = P[:, :, aidx]
            peak = np.unravel_index(int(np.argmax(panel)), panel.shape)
            print(
                f"{vname:<14}{dname:<8}{panel.mean():>8.3f}{panel.max():>8.3f}"
                f"{str((int(peak[0]), int(peak[1]))):>10}"
            )

    print("\nDominant direction by mean probability:")
    for vname, P, goal in (
        ("holding=False", P_free, object_rc),
        ("holding=True", P_hold, target_rc),
    ):
        means = {dname: float(P[:, :, aidx].mean()) for dname, _a, aidx in cfg["directions"]}
        ranked = sorted(means.items(), key=lambda kv: -kv[1])
        top = ", ".join(f"{d} {p:.3f}" for d, p in ranked[:2])
        print(f"  {vname:<14} {top}   (sub-goal is {tuple(goal)})")

    # For every cell, take the greedy direction, apply that step, and ask whether the
    # Manhattan distance to the current sub-goal went down. The sub-goal cell itself is
    # excluded, since no move can improve on already being there.
    step_delta = {
        env.UP: (-1, 0), env.DOWN: (+1, 0), env.LEFT: (0, -1), env.RIGHT: (0, +1),
    }
    print("\nGreedy move vs distance to the current sub-goal:")
    for vname, P, goal in (
        ("holding=False", P_free, object_rc),
        ("holding=True", P_hold, target_rc),
    ):
        goal = tuple(int(v) for v in goal)
        improved = total = 0
        wrong_cells = []
        for r in range(g):
            for c in range(g):
                if (r, c) == goal:
                    continue          # already there; no move helps
                best = dir_idx[int(np.argmax(P[r, c, dir_idx]))]
                dr, dc = step_delta[best]
                nr = int(np.clip(r + dr, 0, g - 1))
                nc = int(np.clip(c + dc, 0, g - 1))
                before = abs(r - goal[0]) + abs(c - goal[1])
                after = abs(nr - goal[0]) + abs(nc - goal[1])
                total += 1
                if after < before:
                    improved += 1
                else:
                    wrong_cells.append((r, c))
        pct = 100.0 * improved / max(total, 1)
        print(f"  {vname:<14} {improved}/{total} cells move closer  ({pct:.0f}%)")
        if wrong_cells:
            print(f"                 cells that do not: {wrong_cells}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    g = args.grid_size
    object_rc = (0, 4)
    target_rc = (4, 0)
    for name, rc in (("--object", object_rc), ("--target", target_rc)):
        if not all(0 <= v < g for v in rc):
            raise SystemExit(f"{name}={rc} is outside a {g}x{g} grid.")
    if object_rc == target_rc:
        raise SystemExit("--object and --target must differ.")

    checkpoint = resolve_checkpoint(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PickAndPlaceEnv(grid_size=g)
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)

    agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim)
    agent.load(checkpoint)
    agent.network.to(device).eval()

    env.reset(seed=0)   # allocates agent_pos / object_pos / target_pos, which start as None
    build_observation = make_observation_builder(env, object_rc, target_rc)

    V_free, P_free = sweep(agent, build_observation, g, device, holding=False)
    V_hold, P_hold = sweep(agent, build_observation, g, device, holding=True)

    # Read direction indices from the env's own constants so a reordered action space
    # cannot silently mislabel the panels.
    cfg = {
        "grid_size": g,
        "object_rc": object_rc,
        "target_rc": target_rc,
        "action_dim": action_dim,
        "checkpoint_name": os.path.basename(checkpoint),
        "shared_scale": not args.independent_scale,
        "extent": (-0.5, g - 0.5, g - 0.5, -0.5),
        "directions": [
            ("UP", "↑", env.UP), ("DOWN", "↓", env.DOWN),
            ("LEFT", "←", env.LEFT), ("RIGHT", "→", env.RIGHT),
        ],
    }

    fig_value = plot_value_maps(V_free, V_hold, cfg)
    fig_diff, diff = plot_difference_map(V_free, V_hold, cfg)
    fig_policy = plot_policy_maps(P_free, P_hold, cfg)

    if not args.no_save:
        outdir = _abs(args.outdir)
        os.makedirs(outdir, exist_ok=True)
        for fig, name in (
            (fig_value, "value_heatmap.png"),
            (fig_diff, "value_difference_heatmap.png"),
            (fig_policy, "policy_heatmaps.png"),
        ):
            fig.savefig(
                os.path.join(outdir, name), dpi=200, bbox_inches="tight", facecolor="white"
            )

    if args.stats:
        print_stats(env, V_free, V_hold, P_free, P_hold, diff, cfg)

    if args.no_show:
        plt.close("all")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
