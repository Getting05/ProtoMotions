"""Render the unchanged P2 reference with retained reconstruction geometry.

The right panel projects the exact same boxes and recorded body origins onto
the travel/height plane. It is a kinematic inspection, never a policy rollout.
"""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .extract_motion import MotionArchive
from .geometry import CollisionModel, sha256


COLORS = [(47, 164, 157), (66, 117, 180)]


def object_color(obj, index):
    return COLORS[index % 2] if obj["kind"] == "stair_tread" else (218, 159, 63)


def vertices(box):
    v = np.array([[x, y, z] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]])
    return (v * box["size"]) @ Rotation.from_quat(box["rotation"]).as_matrix().T + box["translation"]


def visual_model(path, scene, width, height):
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ("meshdir", "texturedir"):
            if compiler.get(attr):
                compiler.set(attr, str((path.parent / compiler.get(attr)).resolve()))
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(width))
    glob.set("offheight", str(height))
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="inspection_ground", type="plane", size="0 0 .05",
                  rgba=".23 .27 .32 1", contype="0", conaffinity="0", group="0")
    ET.SubElement(world, "light", pos="0 0 5", dir="-.2 .1 -1", directional="true", diffuse=".9 .9 .9")
    for i, obj in enumerate(scene["objects"]):
        for j, box in enumerate(obj["primitives"]):
            if box.get("role") == "ground":
                continue
            rgb = np.array(object_color(obj, i)) / 255
            ET.SubElement(world, "geom", name=f"scene_{obj['object_id']}_{j}", type="box",
                size=" ".join(map(str, np.asarray(box["size"]) / 2)),
                pos=" ".join(map(str, box["translation"])),
                quat=" ".join(map(str, np.array(box["rotation"])[[3, 0, 1, 2]])),
                rgba=" ".join(map(str, [*rgb, 1. if obj["kind"] == "stair_tread" else .35])),
                contype="0", conaffinity="0", group="2")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def render(record, root, output, model_path, title):
    scene_path = root / record["scene_file"]
    if sha256(scene_path) != record["scene_sha256"] or sha256(model_path) != record["asset_sha256"]:
        raise ValueError("Scene or model hash differs from the validated reconstruction")
    scene = json.loads(scene_path.read_text())
    quality = json.loads((root / record["quality_file"]).read_text())
    archive = MotionArchive(record["motion_shard"])
    motion = archive.get(record["motion_local_id"])
    if motion.key != record["motion_key"] or motion.digest() != record["source_motion_sha256"]:
        raise ValueError("Source motion differs from reconstruction")
    motion.pos += np.asarray(record["scene_origin_offset"])
    collision = CollisionModel(model_path, 12)
    error = collision.verify_fk(motion)
    points, _, _ = collision.world_samples(motion)
    foot_ids = [collision.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    foot_paths = []
    for b in foot_ids:
        p = points[:, collision.surface.body == b]
        foot_paths.append(p[np.arange(len(p)), p[:, :, 2].argmin(axis=1)])
    with np.load(root / record["contact_file"], allow_pickle=False) as labels:
        states = labels["link_contact_state"].copy()
    width, height = 1280, 800
    viewport_w, viewport_h = 820, 640
    model = visual_model(model_path, scene, viewport_w, viewport_h)
    data = mujoco.MjData(model)
    body_ids = [i for i in range(1, model.nbody) if model.body_jntnum[i] > 0]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in body_ids]
    if names != collision.body_names:
        raise ValueError("Visual and collision body ordering differ")
    parents = [int(model.body_parentid[b]) - 1 for b in body_ids]
    forward = motion.pos[-1, 0] - motion.pos[0, 0]
    forward[2] = 0
    forward /= max(np.linalg.norm(forward), 1e-9)
    if np.linalg.norm(forward) < .5:
        forward = np.array([1., 0., 0.])
    origin = motion.pos[0, 0].copy()
    origin[2] = 0.
    def project(p):
        return np.stack(((p - origin) @ forward, p[..., 2]), axis=-1)
    all_projected = project(motion.pos)
    xmin, xmax = float(all_projected[..., 0].min()) - .25, float(all_projected[..., 0].max()) + .25
    zmin, zmax = -.05, float(all_projected[..., 1].max()) + .2
    x0, x1, y0, y1 = 868, 1250, 665, 160
    def xy(p):
        return (x0 + (p[0] - xmin) / (xmax - xmin) * (x1 - x0),
                y0 - (p[1] - zmin) / (zmax - zmin) * (y0 - y1))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 14)
    tiny = ImageFont.truetype(font_path, 12)
    background = Image.new("RGB", (width, height), (18, 23, 32))
    draw = ImageDraw.Draw(background)
    draw.rectangle((840, 92, width - 1, 735), fill=(245, 247, 250))
    draw.text((860, 106), "RECONSTRUCTED SIDE PROFILE", font=font, fill=(35, 44, 60))
    for z in np.arange(0., zmax, .5):
        a, b = xy((xmin, z)), xy((xmax, z))
        draw.line([a, b], fill=(219, 225, 234))
        draw.text((844, a[1] - 8), f"{z:.1f}", font=tiny, fill=(80, 88, 102))
    for x in np.arange(np.ceil(xmin / .5) * .5, xmax, .5):
        p = xy((x, zmin))
        draw.text((p[0] - 10, y0 + 12), f"{x:.1f}", font=tiny, fill=(80, 88, 102))
    draw.text((940, 712), "Travel (m) / height (m)", font=small, fill=(55, 65, 80))
    for i, obj in enumerate(scene["objects"]):
        for box in obj["primitives"]:
            if box.get("role") == "ground":
                continue
            polygon = project(vertices(box))
            hull = ConvexHull(polygon)
            color = object_color(obj, i)
            if obj["kind"] != "stair_tread":
                color = tuple(int(c * .4 + 245 * .6) for c in color)
            draw.polygon([xy(p) for p in polygon[hull.vertices]], fill=color, outline=object_color(obj, i))
        if obj["kind"] == "stair_tread":
            p = xy(project(np.array(obj["surface_point"])))
            draw.text((p[0] - 12, p[1] - 18), f"{obj['surface_point'][2]:.2f}", font=tiny, fill=(30, 48, 65))
    foot_colors = [(237, 120, 55), (180, 70, 180)]
    for path, color in zip(foot_paths, foot_colors):
        draw.line([xy(p) for p in project(path)], fill=tuple(int(c * .45 + 245 * .55) for c in color), width=1)
    draw.text((860, 137), "L sole: orange   R sole: purple", font=small, fill=(65, 75, 90))
    camera = mujoco.MjvCamera()
    bounds = np.array([motion.pos.min(axis=(0, 1)), motion.pos.max(axis=(0, 1))])
    camera.lookat[:] = bounds.mean(axis=0)
    camera.lookat[2] = max(.7, camera.lookat[2] - .1)
    camera.distance = max(3.4, float(np.ptp(bounds, axis=0).max()) * 1.85)
    camera.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])) + 120)
    camera.elevation = -19
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[[0, 1, 2]] = 1
    stairs = [o for o in scene["objects"] if o["kind"] == "stair_tread"]
    kept_ids = {str(o["object_id"]) for o in scene["objects"]}
    penetration = max([v for k, v in quality["geometry"]["max_penetration_m_by_object"].items()
                       if k in kept_ids] + [0.])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".partial.mp4")
    n = len(motion.pos)
    with mujoco.Renderer(model, height=viewport_h, width=viewport_w) as renderer:
        with imageio.get_writer(str(tmp), fps=1 / motion.dt, codec="libx264", quality=8,
                               macro_block_size=2, ffmpeg_log_level="error") as writer:
            for t in range(n):
                data.qpos[:3] = motion.pos[t, 0]
                data.qpos[3:7] = motion.rot[t, 0, [3, 0, 1, 2]]
                data.qpos[7:] = motion.dof[t]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                frame = background.copy()
                frame.paste(Image.fromarray(renderer.render()), (0, 92))
                d = ImageDraw.Draw(frame)
                d.text((18, 10), f"P2 + RECONSTRUCTED TERRAIN | {title} | motion {motion.local_id}", font=font, fill="white")
                d.text((18, 39), f"{record['status']} | {len(stairs)} retained treads | t={t * motion.dt:.2f}s / {(n - 1) * motion.dt:.2f}s", font=font, fill=(247, 195, 94))
                d.text((18, 65), Path(motion.key).name, font=small, fill=(184, 197, 213))
                pose = project(motion.pos[t])
                for b, parent in enumerate(parents):
                    if parent >= 0:
                        d.line([xy(pose[b]), xy(pose[parent])], fill=(65, 73, 88), width=2)
                for path, body, color in zip(foot_paths, foot_ids, foot_colors):
                    x, y = xy(project(path[t]))
                    d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color if states[t, body] == 1 else "white", outline=color, width=2)
                d.text((18, 748), f"Retained geometry: max penetration {penetration * 1000:.1f} mm | support force residual p95 {quality['support']['force_residual_p95_over_weight']:.2f} body weights", font=small, fill=(196, 207, 222))
                d.text((18, 774), "Teal/blue: inferred treads | amber: other contact hypotheses | kinematic replay, not a controller rollout", font=small, fill=(150, 168, 187))
                writer.append_data(np.asarray(frame))
                if t in (0, n // 2, n - 1):
                    frame.save(output.with_name(output.stem + f"_frame{t:04d}.png"))
                if t == n // 2:
                    frame.save(output.with_suffix(".png"))
                if t % 90 == 0:
                    print(f"{output.stem}: rendered {t + 1}/{n}", flush=True)
    tmp.replace(output)
    metadata = {"motion_id": motion.local_id, "source_key": motion.key, "source_motion_file": str(archive.path),
                "scene_file": str(scene_path), "scene_sha256": sha256(scene_path), "asset_sha256": sha256(model_path),
                "frames": n, "fps": 1 / motion.dt, "fk_max_error_m": error, "retained_treads": len(stairs),
                "status": record["status"], "reasons": record["reasons"], "max_retained_penetration_m": penetration,
                "reference_pose_modified": False, "scene_origin_offset": record["scene_origin_offset"],
                "rendering": "kinematic reference with retained reconstruction; no controller or dynamics rollout",
                "output_sha256": sha256(output)}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--motion-id", type=int, required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Reference inspection")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    record = next(r for r in records if r["motion_local_id"] == args.motion_id)
    render(record, args.manifest.parent.resolve(), args.output.resolve(), args.model.resolve(), args.title)


if __name__ == "__main__":
    main()
