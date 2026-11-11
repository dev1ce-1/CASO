#!/usr/bin/env python
"""Optional offline diagnostics of spatial and semantic risk fields.

Produces the distance field Rd and class weights Rc used by the RiskOcc head,
plus an auxiliary morphological boundary shell Rb. Rb is not a term of the
paper's RiskOcc loss. Formal training computes Rd and Rc online and does not
read these diagnostic files. This utility is not a required training step."""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import yaml

try:
    import scipy.ndimage as ndi
except Exception as exc:  # pragma: no cover - environment guard
    sys.stderr.write("FATAL: scipy.ndimage required (%s)\n" % exc)
    sys.exit(2)

# Class-risk levels (match BEVOCCHead2D_RiskOcc.risk_class_weights).
R_VULNERABLE = 1.0
R_DYNAMIC = 0.6
R_STATIC = 0.3
R_FREE = 0.0

# Quantization codes for compact Rc storage.
RC_LEVELS = np.array([R_FREE, R_STATIC, R_DYNAMIC, R_VULNERABLE], dtype=np.float32)


def load_protocol(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_class_lut(protocol):
    """Return (lut float32[num_classes], free_id). Same groups as the head."""
    names = list(protocol["class_names"])
    free_id = int(protocol["free_id"])
    vuln = set(protocol.get("vulnerable_classes", []))
    dyn = set(protocol.get("dynamic_classes", []))
    # dynamic group in the head excludes the vulnerable agents; mirror that.
    dyn = dyn - vuln
    lut = np.empty(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        if i == free_id or name == "free":
            lut[i] = R_FREE
        elif name in vuln:
            lut[i] = R_VULNERABLE
        elif name in dyn:
            lut[i] = R_DYNAMIC
        else:
            lut[i] = R_STATIC
    return lut, free_id


def build_distance_field(protocol):
    """Replicate the head's distance field exactly (numpy mirror)."""
    pr = [float(v) for v in protocol["pc_range"]]
    vs = float(protocol["voxel_size"])
    distance_max = 40.0          # == head.risk_distance_max
    near = float(protocol.get("d_near", 20.0))
    nx = int(round((pr[3] - pr[0]) / vs))
    ny = int(round((pr[4] - pr[1]) / vs))
    nz = int(round((pr[5] - pr[2]) / vs))
    xs = (np.arange(nx, dtype=np.float32) + 0.5) * vs + pr[0]
    ys = (np.arange(ny, dtype=np.float32) + 0.5) * vs + pr[1]
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    d_bev = np.sqrt(xg ** 2 + yg ** 2).astype(np.float32)
    rd2d = np.clip((distance_max - d_bev) / distance_max, 0.0, 1.0).astype(np.float32)
    rd3d = np.repeat(rd2d[:, :, None], nz, axis=2)
    near_mask2d = d_bev <= near
    return {
        "rd3d": rd3d, "rd2d": rd2d, "d_bev": d_bev,
        "near_mask2d": near_mask2d, "grid": (nx, ny, nz),
        "distance_max": distance_max, "near": near,
    }


def build_blocking_slab(protocol):
    """Collision-relevant slab from safety_protocol: z in z_plan & d_bev<=d_block.

    Returns a (nx, ny, nz) bool mask. Reuses the already-validated protocol
    geometry; Rb is confined to this slab so it stays a thin near-field shell.
    """
    pr = [float(v) for v in protocol["pc_range"]]
    vs = float(protocol["voxel_size"])
    nx = int(round((pr[3] - pr[0]) / vs))
    ny = int(round((pr[4] - pr[1]) / vs))
    nz = int(round((pr[5] - pr[2]) / vs))
    zc = (np.arange(nz, dtype=np.float32) + 0.5) * vs + pr[2]
    z_lo, z_hi = [float(v) for v in protocol["z_plan"]]
    slab_z = (zc >= z_lo) & (zc <= z_hi)                       # (nz,)
    xs = (np.arange(nx, dtype=np.float32) + 0.5) * vs + pr[0]
    ys = (np.arange(ny, dtype=np.float32) + 0.5) * vs + pr[1]
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    d_bev = np.sqrt(xg ** 2 + yg ** 2)
    near_block = d_bev <= float(protocol["d_block"])           # (nx, ny)
    return near_block[:, :, None] & slab_z[None, None, :]


_BOUNDARY_STRUCT = ndi.generate_binary_structure(3, 1)  # 6-connectivity


def boundary_shell(occupied):
    """1-voxel internal boundary: occupied voxels adjacent to non-occupied."""
    eroded = ndi.binary_erosion(occupied, structure=_BOUNDARY_STRUCT,
                                border_value=0)
    return occupied & (~eroded)


def quantize_rc(rc):
    """Map continuous Rc to level codes (uint8) for compact storage."""
    out = np.zeros(rc.shape, dtype=np.uint8)
    for code, level in enumerate(RC_LEVELS):
        out[np.isclose(rc, level)] = code
    return out


def frame_key(info):
    op = info["occ_path"]
    parts = [p for p in op.replace("\\", "/").split("/") if p]
    scene = next((p for p in parts if p.startswith("scene-")), "scene-unknown")
    token = info.get("token") or parts[-1]
    return scene, token


def load_infos(pkl_path, split, frames):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
    if frames is not None and frames > 0:
        infos = infos[:frames]
    return infos


def load_semantics(info):
    op = info["occ_path"]
    path = os.path.join(op, "labels.npz")
    npz = np.load(path)
    return npz["semantics"], npz["mask_camera"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="configs/eval/safety_protocol.yaml")
    ap.add_argument("--train-pkl",
                    default="data/nuscenes/bevdetv2-nuscenes_infos_train.pkl")
    ap.add_argument("--val-pkl",
                    default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl")
    ap.add_argument("--split", choices=["train", "val", "both"], default="train")
    ap.add_argument("--frames", type=int, default=0,
                    help="limit to first N frames per split (0 = all)")
    ap.add_argument("--out", default="risk_field/")
    ap.add_argument("--selfcheck", action="store_true",
                    help="print/assert the Stage-0 risk-field assertions")
    ap.add_argument("--no-store", action="store_true",
                    help="compute + self-check only, do not write per-frame files")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    lut, free_id = build_class_lut(protocol)
    df = build_distance_field(protocol)
    slab_mask = build_blocking_slab(protocol)
    nx, ny, nz = df["grid"]

    os.makedirs(args.out, exist_ok=True)
    # Rd is grid-invariant: store once.
    rd_path = os.path.join(args.out, "Rd.npy")
    if not args.no_store:
        np.save(rd_path, df["rd3d"])
    meta = {
        "grid": [nx, ny, nz],
        "pc_range": [float(v) for v in protocol["pc_range"]],
        "voxel_size": float(protocol["voxel_size"]),
        "distance_max": df["distance_max"], "near": df["near"],
        "free_id": free_id,
        "rc_levels": RC_LEVELS.tolist(),
        "rc_level_meaning": {"0": "free=0.0", "1": "static=0.3",
                             "2": "dynamic=0.6", "3": "vulnerable=1.0"},
        "class_lut": lut.tolist(),
        "rb_def": "internal boundary of GT occupied, restricted to blocking "
                  "slab z in z_plan & d_bev<=d_block",
        "z_plan": [float(v) for v in protocol["z_plan"]],
        "d_block": float(protocol["d_block"]),
        "note": "online RiskOcc head does not read these files; gate+Phase2 only",
    }
    if not args.no_store:
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    vuln_ids = [i for i, n in enumerate(protocol["class_names"])
                if n in set(protocol.get("vulnerable_classes", []))]

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", args.train_pkl))
    if args.split in ("val", "both"):
        splits.append(("val", args.val_pkl))

    # Self-check accumulators.
    rd_near = float(df["rd3d"][:, :, 0][df["d_bev"] <= df["near"]].mean())
    rd_far = float(df["rd3d"][:, :, 0][df["d_bev"] >= 30.0].mean())
    a2_violations = 0      # vulnerable voxels with Rc != 1.0
    a2_vuln_voxels = 0
    shell_fracs = []       # shell / within-slab occupied, per frame (diagnostic)
    shell_fracs_all = []   # shell / all voxels, per frame (A3 metric)
    n_done = 0
    t0 = time.time()

    for split_name, pkl_path in splits:
        infos = load_infos(pkl_path, split_name, args.frames)
        for info in infos:
            try:
                sem, mask_cam = load_semantics(info)
            except Exception as exc:
                sys.stderr.write("WARN load failed %s: %s\n"
                                 % (info.get("occ_path"), exc))
                continue
            sem = sem.astype(np.int64)
            mask_cam = mask_cam.astype(bool)

            # Rc: class-risk lookup.
            rc = lut[sem]

            # Rb: 1-voxel internal boundary of occupied, confined to the
            # collision-relevant blocking slab (near-field obstacle front).
            occupied = sem != free_id
            shell = boundary_shell(occupied) & slab_mask

            if args.selfcheck:
                # A2: every vulnerable voxel must carry Rc == 1.0.
                for vid in vuln_ids:
                    m = sem == vid
                    cnt = int(m.sum())
                    if cnt:
                        a2_vuln_voxels += cnt
                        a2_violations += int(np.count_nonzero(
                            ~np.isclose(rc[m], R_VULNERABLE)))
                # A3: shell fraction over all voxels (thin-shell metric).
                shell_n = int(shell.sum())
                shell_fracs_all.append(shell_n / sem.size)
                occ_slab = int((occupied & slab_mask).sum())
                if occ_slab:
                    shell_fracs.append(shell_n / occ_slab)

            if not args.no_store:
                scene, token = frame_key(info)
                fdir = os.path.join(args.out, "frames", scene)
                os.makedirs(fdir, exist_ok=True)
                rc_q = quantize_rc(rc)
                rb_coords = np.argwhere(shell).astype(np.uint16)
                np.savez_compressed(os.path.join(fdir, token + ".npz"),
                                    rc=rc_q, rb_coords=rb_coords,
                                    grid=np.array([nx, ny, nz], dtype=np.int32))
            n_done += 1

    elapsed = time.time() - t0
    summary = {
        "frames": n_done,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_frame": round(elapsed / max(n_done, 1), 4),
        "out": args.out,
    }
    print("[precompute] %s" % json.dumps(summary))

    if args.selfcheck:
        a1_pass = rd_near > rd_far
        a2_pass = (a2_vuln_voxels > 0) and (a2_violations == 0)
        max_shell = max(shell_fracs) if shell_fracs else 0.0
        mean_shell = float(np.mean(shell_fracs)) if shell_fracs else 0.0
        max_shell_all = max(shell_fracs_all) if shell_fracs_all else 0.0
        mean_shell_all = float(np.mean(shell_fracs_all)) if shell_fracs_all else 0.0
        a3_pass = max_shell_all < 0.05
        print("[selfcheck] A1 Rd monotone : near_mean=%.4f far_mean=%.4f -> %s"
              % (rd_near, rd_far, "PASS" if a1_pass else "FAIL"))
        print("[selfcheck] A2 Rc hits VRU : vuln_voxels=%d violations=%d -> %s"
              % (a2_vuln_voxels, a2_violations, "PASS" if a2_pass else "FAIL"))
        print("[selfcheck] A3 Rb thin     : shell/allvox mean=%.4f max=%.4f "
              "(<0.05) | shell/slab_occ mean=%.4f -> %s"
              % (mean_shell_all, max_shell_all, mean_shell,
                 "PASS" if a3_pass else "FAIL"))
        result = {"A1_rd_monotone": bool(a1_pass),
                  "A2_rc_hits_vru": bool(a2_pass),
                  "A3_rb_thin_shell": bool(a3_pass),
                  "rd_near_mean": rd_near, "rd_far_mean": rd_far,
                  "a2_vuln_voxels": a2_vuln_voxels,
                  "a2_violations": a2_violations,
                  "shell_frac_allvox_mean": mean_shell_all,
                  "shell_frac_allvox_max": max_shell_all,
                  "shell_frac_slabocc_mean": mean_shell,
                  "frames_checked": n_done}
        with open(os.path.join(args.out, "selfcheck.json"), "w") as f:
            json.dump(result, f, indent=2)
        all_pass = a1_pass and a2_pass and a3_pass
        print("[selfcheck] OVERALL -> %s" % ("PASS" if all_pass else "FAIL"))
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
