"""Lightweight parser for CMU Mocap .asf (skeleton) and .amc (motion) files.

Layer 1 of the CMU Mocap conversion pipeline.
Parses raw ASF/AMC files into structured Python dataclasses, no external
dependencies beyond numpy.

Usage::

    from cmu_parser import parse_asf, parse_amc

    skeleton = parse_asf("data/subject_07/07.asf")
    motion = parse_amc("data/subject_07/07_01.amc", skeleton)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BoneDef:
    """Definition of a single bone parsed from :bonedata."""

    id: int
    name: str
    direction: np.ndarray  # (3,)
    length: float
    axis: np.ndarray  # (3,)
    axis_order: str  # e.g. "XYZ"
    dof: list[str] = field(default_factory=list)
    limits: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class SkeletonDef:
    """Complete skeleton parsed from an .asf file."""

    bones: list[BoneDef]
    hierarchy: dict[str, list[str]]
    root_order: list[str]
    root_axis: str
    angle_unit: str  # "deg" or "rad"

    @property
    def bone_by_name(self) -> dict[str, BoneDef]:
        return {b.name: b for b in self.bones}

    @property
    def dof_bones(self) -> list[BoneDef]:
        """Bones that have at least one DOF, sorted by bone id."""
        return sorted([b for b in self.bones if b.dof], key=lambda b: b.id)

    @property
    def total_dofs(self) -> int:
        return sum(len(b.dof) for b in self.dof_bones)

    def joint_names(self) -> list[str]:
        """Stable per-DOF names in the form ``bone_dof`` (e.g. ``lfemur_rx``)."""
        names: list[str] = []
        for bone in self.dof_bones:
            for d in bone.dof:
                names.append(f"{bone.name}_{d}")
        return names


@dataclass
class MotionData:
    """Motion data parsed from an .amc file."""

    frames: list[dict[str, list[float]]]
    root_data: np.ndarray  # (T, 6)
    frame_ids: list[int]
    num_frames: int
    fps: float
    angle_unit: str  # "deg" or "rad"


def _tokenize_line(line: str) -> list[str]:
    return line.split()


def _parse_bone_block(lines: list[str]) -> BoneDef:
    """Parse a single ``begin ... end`` block inside ``:bonedata``."""
    bone_id = -1
    name = ""
    direction = np.zeros(3)
    length = 0.0
    axis = np.zeros(3)
    axis_order = "XYZ"
    dof: list[str] = []
    limits: list[tuple[float, float]] = []

    in_limits = False
    for raw in lines:
        tokens = _tokenize_line(raw)
        if not tokens:
            continue
        key = tokens[0].lower()

        if key == "id":
            bone_id = int(tokens[1])
        elif key == "name":
            name = tokens[1]
        elif key == "direction":
            direction = np.array([float(v) for v in tokens[1:4]])
        elif key == "length":
            length = float(tokens[1])
        elif key == "axis":
            axis = np.array([float(_safe_float(v)) for v in tokens[1:4]])
            axis_order = tokens[4] if len(tokens) > 4 else "XYZ"
        elif key == "dof":
            dof = tokens[1:]
            in_limits = False
        elif key == "limits":
            in_limits = True
            limits.extend(_parse_limit_tokens(tokens[1:]))
        elif in_limits and key.startswith("("):
            in_limits = True
            limits.extend(_parse_limit_tokens(tokens))

    return BoneDef(
        id=bone_id,
        name=name,
        direction=direction,
        length=length,
        axis=axis,
        axis_order=axis_order,
        dof=dof,
        limits=limits,
    )


def _safe_float(s: str) -> float:
    """Parse a float that may use ``e-015`` style exponents."""
    try:
        return float(s)
    except ValueError:
        return float(s.replace("e-0", "e-").replace("E-0", "E-") if "e-0" in s.lower() else s)


def _parse_limit_tokens(tokens: list[str]) -> list[tuple[float, float]]:
    """Parse limit tokens like ``( -160.0 20.0 )`` spread across lines."""
    results: list[tuple[float, float]] = []
    cleaned = " ".join(tokens)
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    parts = cleaned.split()
    for i in range(0, len(parts) - 1, 2):
        try:
            results.append((float(parts[i]), float(parts[i + 1])))
        except (ValueError, IndexError):
            break
    return results


def _parse_hierarchy(lines: list[str]) -> dict[str, list[str]]:
    """Parse ``:hierarchy`` section into ``{parent: [children]}``."""
    hierarchy: dict[str, list[str]] = {}
    for raw in lines:
        tokens = _tokenize_line(raw)
        if not tokens:
            continue
        parent = tokens[0]
        children = tokens[1:]
        hierarchy[parent] = hierarchy.get(parent, []) + children
    return hierarchy


def parse_asf(path: str | Path) -> SkeletonDef:
    """Parse a CMU ``.asf`` skeleton file.

    Args:
        path: Path to the .asf file.

    Returns:
        A SkeletonDef with all bone definitions and hierarchy.
    """
    path = Path(path)
    with open(path) as f:
        content = f.read()

    lines = content.splitlines()
    n = len(lines)

    root_order: list[str] = []
    root_axis = "XYZ"
    angle_unit = "deg"
    bones: list[BoneDef] = []
    hierarchy: dict[str, list[str]] = {}

    section = ""
    bone_block_lines: list[str] = []
    hierarchy_lines: list[str] = []
    inside_bone_block = False

    i = 0
    while i < n:
        stripped = lines[i].strip()

        # Skip comments
        if stripped.startswith("#") or not stripped:
            i += 1
            continue

        # Section headers
        if stripped == ":units":
            section = "units"
            i += 1
            continue
        elif stripped == ":root":
            section = "root"
            i += 1
            continue
        elif stripped == ":bonedata":
            section = "bonedata"
            i += 1
            continue
        elif stripped == ":hierarchy":
            section = "hierarchy"
            i += 1
            continue
        elif stripped.startswith(":version") or stripped.startswith(":name") or stripped.startswith(":documentation"):
            section = "other"
            i += 1
            continue

        # Parse based on section
        if section == "units":
            if stripped.startswith("angle"):
                angle_unit = stripped.split()[1].strip()

        elif section == "root":
            tokens = _tokenize_line(stripped)
            if tokens[0] == "order":
                root_order = tokens[1:]
            elif tokens[0] == "axis":
                root_axis = tokens[1]

        elif section == "bonedata":
            if stripped == "begin":
                inside_bone_block = True
                bone_block_lines = []
            elif stripped == "end":
                inside_bone_block = False
                bones.append(_parse_bone_block(bone_block_lines))
            elif inside_bone_block:
                bone_block_lines.append(lines[i])

        elif section == "hierarchy":
            if stripped == "begin":
                hierarchy_lines = []
            elif stripped == "end":
                hierarchy = _parse_hierarchy(hierarchy_lines)
            else:
                hierarchy_lines.append(lines[i])

        i += 1

    return SkeletonDef(
        bones=bones,
        hierarchy=hierarchy,
        root_order=root_order,
        root_axis=root_axis,
        angle_unit=angle_unit,
    )


def parse_amc(path: str | Path, skeleton: SkeletonDef, fps: float = 120.0) -> MotionData:
    """Parse a CMU ``.amc`` motion file.

    Args:
        path: Path to the .amc file.
        skeleton: The corresponding parsed skeleton.
        fps: Frame rate to assume (CMU default is 120 Hz).

    Returns:
        A MotionData with per-frame joint values.
    """
    path = Path(path)

    # Detect angle unit from header
    angle_unit = "deg"
    with open(path) as f:
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            if ":DEGREES" in line:
                angle_unit = "deg"
            elif ":RADIANS" in line:
                angle_unit = "rad"

    frames: list[dict[str, list[float]]] = []
    root_data_rows: list[list[float]] = []
    frame_ids: list[int] = []

    current_frame: dict[str, list[float]] | None = None
    current_root: list[float] = []
    current_frame_id = -1

    bone_dof_counts = {b.name: len(b.dof) for b in skeleton.bones}

    with open(path) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith(":"):
                continue

            # Frame number line: an integer
            if stripped[0].isdigit() or (stripped[0] == "-" and stripped[1:].replace(".", "", 1).isdigit()):
                # Save previous frame
                if current_frame is not None:
                    frames.append(current_frame)
                    root_data_rows.append(current_root)
                    frame_ids.append(current_frame_id)

                # Try parsing as frame id (integer)
                try:
                    current_frame_id = int(float(stripped))
                    current_frame = {}
                    current_root = []
                except ValueError:
                    continue
                continue

            tokens = _tokenize_line(stripped)
            if not tokens:
                continue

            bone_name = tokens[0]
            values = [_safe_float(v) for v in tokens[1:]]

            if bone_name == "root":
                current_root = values[:6]
            else:
                if current_frame is not None:
                    expected = bone_dof_counts.get(bone_name, 0)
                    if expected == 0:
                        continue
                    if len(values) != expected:
                        pass  # best-effort: store what we have
                    current_frame[bone_name] = values

    # Save last frame
    if current_frame is not None:
        frames.append(current_frame)
        root_data_rows.append(current_root)
        frame_ids.append(current_frame_id)

    root_data = np.array(root_data_rows, dtype=np.float64) if root_data_rows else np.zeros((0, 6))

    return MotionData(
        frames=frames,
        root_data=root_data,
        frame_ids=frame_ids,
        num_frames=len(frames),
        fps=fps,
        angle_unit=angle_unit,
    )
