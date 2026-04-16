"""Batch-convert, inspect, and summarize CMU Mocap data.

Three sub-commands:

    convert   -- .asf + .amc  ->  .npz  (batch conversion)
    inspect   -- inspect a single .npz file
    summarize -- summarize all .npz files in a directory

Usage::

    # Batch convert
    python scripts/mocap/batch_convert_cmu.py convert \\
        --input_dir data/human_gait/subject_07 \\
        --output_dir data/human_gait/processed/cmu_subject_07 \\
        --subject_id subject_07

    # Inspect a single .npz
    python scripts/mocap/batch_convert_cmu.py inspect \\
        data/human_gait/processed/cmu_subject_07/07_01.npz

    # Summarize a subject directory
    python scripts/mocap/batch_convert_cmu.py summarize \\
        data/human_gait/processed/cmu_subject_07

Backward compatibility -- running without a sub-command defaults to ``convert``::

    python scripts/mocap/batch_convert_cmu.py \\
        --input_dir data/human_gait/subject_07 \\
        --output_dir data/human_gait/processed/cmu_subject_07
"""

from __future__ import annotations

import argparse
import json
import logging
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmu_parser import parse_amc, parse_asf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Convert  (.asf + .amc -> .npz)
# ---------------------------------------------------------------------------


def _find_files(input_dir: Path) -> tuple[Path, list[Path]]:
    """Locate the .asf skeleton and all .amc motion files."""
    asf_files = sorted(input_dir.glob("*.asf"))
    amc_files = sorted(input_dir.glob("*.amc"))

    if not asf_files:
        raise FileNotFoundError(f"No .asf file found in {input_dir}")
    if not amc_files:
        raise FileNotFoundError(f"No .amc files found in {input_dir}")

    return asf_files[0], amc_files


def _motion_to_arrays(
    motion,
    skeleton,
    deg_to_rad: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert parsed MotionData to standardized numpy arrays.

    Returns:
        angles: (T, J) float64 joint angles in rad
        root_translation: (T, 3) float64
        root_rotation: (T, 3) float64 euler angles in rad
        frame_ids: (T,) int32
    """
    dof_bones = skeleton.dof_bones
    joint_names = skeleton.joint_names()
    total_dofs = len(joint_names)

    num_frames = motion.num_frames
    angles = np.zeros((num_frames, total_dofs), dtype=np.float64)

    for t, frame_dict in enumerate(motion.frames):
        col = 0
        for bone in dof_bones:
            n_dof = len(bone.dof)
            values = frame_dict.get(bone.name, [0.0] * n_dof)
            if len(values) < n_dof:
                values = values + [0.0] * (n_dof - len(values))
            angles[t, col : col + n_dof] = values[:n_dof]
            col += n_dof

    if deg_to_rad:
        angles = np.deg2rad(angles)

    root_data = motion.root_data.copy()
    root_translation = root_data[:, :3]

    root_rotation = root_data[:, 3:6].copy()
    if deg_to_rad:
        root_rotation = np.deg2rad(root_rotation)

    frame_ids = np.array(motion.frame_ids, dtype=np.int32)

    return angles, root_translation, root_rotation, frame_ids


def convert_single_amc(
    asf_path: Path,
    amc_path: Path,
    output_dir: Path,
    subject_id: str,
    fps: float,
    skeleton,
) -> dict:
    """Convert one .amc file to .npz.

    Returns:
        Summary dict for this conversion.
    """
    motion_name = amc_path.stem
    output_path = output_dir / f"{motion_name}.npz"

    log.info("Parsing %s ...", amc_path.name)
    motion = parse_amc(amc_path, skeleton, fps=fps)

    deg_to_rad = motion.angle_unit == "deg"
    angles, root_translation, root_rotation, frame_ids = _motion_to_arrays(motion, skeleton, deg_to_rad=deg_to_rad)

    joint_names = np.array(skeleton.joint_names())

    metadata = {
        "parser_version": "1.0.0",
        "source_angle_unit": motion.angle_unit,
        "output_angle_unit": "rad",
        "conversion": "deg2rad" if deg_to_rad else "none",
        "scale_unit": "cm (AMC raw, no scaling applied)",
        "fps_source": "default (not reliably obtainable from file)",
        "num_dofs": int(angles.shape[1]),
        "has_nan": bool(np.any(np.isnan(angles))),
    }

    np.savez(
        str(output_path),
        angles=angles,
        joint_names=joint_names,
        num_frames=np.int32(motion.num_frames),
        fps=np.float64(fps),
        angle_unit="rad",
        root_translation=root_translation,
        root_rotation=root_rotation,
        frame_ids=frame_ids,
        source_asf=str(asf_path),
        source_amc=str(amc_path),
        subject_id=subject_id,
        motion_name=motion_name,
        metadata_json=json.dumps(metadata, indent=2),
    )

    return {
        "source_amc": str(amc_path),
        "output_path": str(output_path),
        "num_frames": int(motion.num_frames),
        "num_joints": int(angles.shape[1]),
        "success": True,
        "error": None,
    }


def batch_convert(
    input_dir: str | Path,
    output_dir: str | Path,
    subject_id: str,
    fps: float = 120.0,
) -> dict:
    """Run batch conversion for all .amc files in a subject directory.

    Returns:
        Summary dict with per-file results.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    asf_path, amc_files = _find_files(input_dir)
    log.info("Found skeleton: %s", asf_path.name)
    log.info("Found %d motion files", len(amc_files))

    skeleton = parse_asf(asf_path)
    log.info(
        "Skeleton: %d bones, %d DOF-bearing bones, %d total DOFs",
        len(skeleton.bones),
        len(skeleton.dof_bones),
        skeleton.total_dofs,
    )
    log.info("Joint columns: %s", " | ".join(skeleton.joint_names()))

    results: list[dict] = []
    for amc_path in amc_files:
        try:
            result = convert_single_amc(
                asf_path=asf_path,
                amc_path=amc_path,
                output_dir=output_dir,
                subject_id=subject_id,
                fps=fps,
                skeleton=skeleton,
            )
        except Exception as exc:
            log.error("FAILED %s: %s", amc_path.name, exc, exc_info=True)
            results.append(
                {
                    "source_amc": str(amc_path),
                    "output_path": None,
                    "num_frames": 0,
                    "num_joints": 0,
                    "success": False,
                    "error": str(exc),
                }
            )
            continue
        log.info(
            "  OK  %s  ->  %s  (%d frames, %d joints)",
            amc_path.name,
            result["output_path"],
            result["num_frames"],
            result["num_joints"],
        )
        results.append(result)

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    summary = {
        "subject_id": subject_id,
        "source_directory": str(input_dir),
        "output_directory": str(output_dir),
        "skeleton_file": str(asf_path),
        "total_amc_files": len(amc_files),
        "successful_conversions": success_count,
        "failed_conversions": fail_count,
        "fps": fps,
        "joint_names": skeleton.joint_names(),
        "total_dofs": skeleton.total_dofs,
        "files": results,
    }

    summary_path = output_dir / "conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Summary written to %s", summary_path)
    log.info("Done: %d/%d converted, %d failed", success_count, len(amc_files), fail_count)

    return summary


def run_convert(args: argparse.Namespace) -> None:
    """Execute the batch convert step."""
    print("\n" + "=" * 60)
    print("  Step 1: Batch Convert ASF+AMC -> NPZ")
    print("=" * 60)

    start = time.time()
    summary = batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        subject_id=args.subject_id,
        fps=args.fps,
    )
    elapsed = time.time() - start
    log.info("Total time: %.1f seconds", elapsed)

    if summary["failed_conversions"] > 0:
        sys.exit(1)

    print("\n[DONE] Convert complete.")


# ---------------------------------------------------------------------------
# Step 2: Inspect  (single .npz)
# ---------------------------------------------------------------------------


def run_inspect(args: argparse.Namespace) -> None:
    """Inspect a single .npz file."""
    path = Path(args.npz_file)
    show_frames = args.show_frames

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    data = np.load(path, allow_pickle=False)

    print("\n" + "=" * 60)
    print(f"  Step 2: Inspect {path.name}")
    print("=" * 60)

    angles = data["angles"]
    joint_names = data["joint_names"]
    num_frames = int(data["num_frames"])
    fps = float(data["fps"])
    angle_unit = str(data["angle_unit"])
    subject_id = str(data["subject_id"])
    motion_name = str(data["motion_name"])

    print(f"\n  subject_id    : {subject_id}")
    print(f"  motion_name   : {motion_name}")
    print(f"  angles.shape  : {angles.shape}")
    print(f"  num_frames    : {num_frames}")
    print(f"  fps           : {fps}")
    print(f"  angle_unit    : {angle_unit}")
    print(f"  num_joints    : {len(joint_names)}")

    has_nan = bool(np.any(np.isnan(angles)))
    has_inf = bool(np.any(np.isinf(angles)))
    print(f"  has NaN       : {has_nan}")
    print(f"  has Inf       : {has_inf}")

    if "root_translation" in data:
        rt = data["root_translation"]
        print(f"  root_translation.shape : {rt.shape}")
        print(f"    range X: [{rt[:, 0].min():.2f}, {rt[:, 0].max():.2f}]")
        print(f"    range Y: [{rt[:, 1].min():.2f}, {rt[:, 1].max():.2f}]")
        print(f"    range Z: [{rt[:, 2].min():.2f}, {rt[:, 2].max():.2f}]")

    if "root_rotation" in data:
        rr = data["root_rotation"]
        print(f"  root_rotation.shape    : {rr.shape}")

    if "frame_ids" in data:
        fid = data["frame_ids"]
        print(f"  frame_ids range: [{fid[0]}, ..., {fid[-1]}]  (len={len(fid)})")

    if "metadata_json" in data:
        meta = json.loads(str(data["metadata_json"]))
        print("\n  metadata:")
        for k, v in meta.items():
            print(f"    {k}: {v}")

    print(f"\n  --- First {show_frames} frames (angles) ---")
    n_show = min(show_frames, angles.shape[0])
    for t in range(n_show):
        vals = angles[t]
        nonzero = [(joint_names[j], f"{vals[j]:.4f}") for j in range(len(vals)) if abs(vals[j]) > 0.01]
        print(f"    frame {t}: {', '.join(f'{n}={v}' for n, v in nonzero[:10])}{'...' if len(nonzero) > 10 else ''}")

    print("\n  --- Angle statistics per DOF ---")
    for j, name in enumerate(joint_names):
        col = angles[:, j]
        print(
            f"    {name:20s}  min={col.min():+8.4f}  max={col.max():+8.4f}  "
            f"mean={col.mean():+8.4f}  std={col.std():8.4f}"
        )

    print("\n[DONE] Inspect complete.")


# ---------------------------------------------------------------------------
# Step 3: Summarize  (whole directory)
# ---------------------------------------------------------------------------


def run_summarize(args: argparse.Namespace) -> None:
    """Summarize all .npz files in a processed directory."""
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"Directory not found: {processed_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Step 3: Summarize Subject")
    print("=" * 60)

    summary_path = processed_dir / "conversion_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print("\n  === conversion_summary.json ===")
        print(f"  Subject ID       : {summary.get('subject_id')}")
        print(f"  Total AMC files  : {summary.get('total_amc_files')}")
        print(f"  Successful       : {summary.get('successful_conversions')}")
        print(f"  Failed           : {summary.get('failed_conversions')}")
        print(f"  FPS              : {summary.get('fps')}")
        print(f"  Total DOFs       : {summary.get('total_dofs')}")
        print(f"  Joint names      : {summary.get('joint_names')}")
        print()

    npz_files = sorted(processed_dir.glob("*.npz"))
    if not npz_files:
        print("  No .npz files found.")
        return

    print(f"  === Inspecting {len(npz_files)} .npz files ===\n")

    all_joint_counts: set[int] = set()
    frame_counts: list[int] = []
    joint_name_lists: list[tuple[str, tuple[str, ...]]] = []
    failed: list[str] = []

    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=False)
            angles = data["angles"]
            jnames = tuple(str(n) for n in data["joint_names"])
            nf = int(data["num_frames"])

            all_joint_counts.add(angles.shape[1])
            frame_counts.append(nf)
            joint_name_lists.append((npz_path.name, jnames))

            has_nan = bool(np.any(np.isnan(angles)))
            print(
                f"    {npz_path.name:20s}  frames={nf:6d}  joints={angles.shape[1]:3d}  "
                f"NaN={has_nan}  shape={angles.shape}"
            )
        except Exception as exc:
            print(f"    {npz_path.name:20s}  ERROR: {exc}")
            failed.append(npz_path.name)

    print("\n  --- Summary ---")
    print(f"  Total npz files : {len(npz_files)}")
    if frame_counts:
        print(f"  Frame count range : [{min(frame_counts)}, {max(frame_counts)}]")
    print(f"  Joint counts found: {all_joint_counts}")
    print(f"  Joint count consistent: {len(all_joint_counts) == 1}")

    if joint_name_lists:
        _, ref_jnames = joint_name_lists[0]
        all_consistent = all(jnames == ref_jnames for _, jnames in joint_name_lists)
        print(f"  Joint name ordering consistent: {all_consistent}")
        if not all_consistent:
            for fname, jnames in joint_name_lists:
                if jnames != ref_jnames:
                    print(f"    MISMATCH: {fname}")

    if failed:
        print(f"\n  Failed files ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")

    if summary_path.exists():
        summary_failures = [f for f in summary.get("files", []) if not f["success"]]
        if summary_failures:
            print(f"\n  Failures recorded in summary ({len(summary_failures)}):")
            for f in summary_failures:
                print(f"    - {Path(f['source_amc']).name}: {f['error']}")

    print("\n[DONE] Summarize complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-convert, inspect, and summarize CMU Mocap data.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- convert ---
    p_convert = subparsers.add_parser("convert", help="Batch-convert .asf+.amc to .npz")
    p_convert.add_argument(
        "--input_dir",
        type=str,
        default="data/human_gait/subject_07",
        help="Directory containing .asf and .amc files",
    )
    p_convert.add_argument(
        "--output_dir",
        type=str,
        default="data/human_gait/processed/cmu_subject_07",
        help="Output directory for .npz files",
    )
    p_convert.add_argument(
        "--subject_id",
        type=str,
        default="subject_07",
        help="Subject identifier (e.g. subject_07)",
    )
    p_convert.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="Frame rate to assume (default: 120 Hz, CMU standard)",
    )

    # --- inspect ---
    p_inspect = subparsers.add_parser("inspect", help="Inspect a single .npz file")
    p_inspect.add_argument("npz_file", type=str, help="Path to the .npz file to inspect.")
    p_inspect.add_argument("--show_frames", type=int, default=5, help="Number of frames to print (default: 5)")

    # --- summarize ---
    p_summarize = subparsers.add_parser("summarize", help="Summarize all .npz files in a directory")
    p_summarize.add_argument("processed_dir", type=str, help="Directory containing .npz files")

    # Backward compatibility: if no sub-command given, default to "convert"
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-") or argv[0] not in ("convert", "inspect", "summarize"):
        argv = ["convert"] + argv

    args = parser.parse_args(argv)

    if args.command == "convert":
        run_convert(args)
    elif args.command == "inspect":
        run_inspect(args)
    elif args.command == "summarize":
        run_summarize(args)


if __name__ == "__main__":
    main()
