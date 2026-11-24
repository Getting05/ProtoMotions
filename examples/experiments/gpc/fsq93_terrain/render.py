"""Render the configured colliders and actual P2 reset pose with a moving camera."""
import argparse
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from p2drift.contracts import MOTOR_JOINT_NAMES

from .runtime import face_goal, reset_cases
from .scene import generate, load_config, scene_asset


WIDTH, HEIGHT = 1280, 720
TOP, BOTTOM = 100, 87
VIEW_HEIGHT = HEIGHT - TOP - BOTTOM
FPS, SECONDS = 24, 6
ACCENT = "#66e8ef"
GOAL = "#ffcf6b"
TEXT = "#f2f6f8"
MUTED = "#a9bdca"
TITLE = {
    "platform_2x2": "Platform / 2 x 2",
    "platform_giant": "Platform / Giant",
    "barrier": "Barrier / Low clearance",
}


@lru_cache
def font(size, bold=False):
    name = f"DejaVuSans{'-Bold' if bold else ''}.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def geometry(spec, kind, pos, size, rgba):
    return spec.worldbody.add_geom(
        type=kind, pos=pos, size=size, rgba=rgba, contype=0, conaffinity=0, group=0
    )


def marker(spec, pt, rgba):
    x, y, z = pt
    for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        pos = (x + 0.31 * np.cos(theta), y + 0.31 * np.sin(theta), z + 0.027)
        geometry(spec, mujoco.mjtGeom.mjGEOM_SPHERE, pos, [0.036, 0., 0.], rgba)
    geometry(spec, mujoco.mjtGeom.mjGEOM_CYLINDER, (x, y, z + 0.017),
             [0.075, 0.015, 0.], rgba)


def guides(spec, scene):
    barrier = scene.kind == "barrier"
    if barrier:
        n = scene.floors[0][2]
        for sign in (-1, 1):
            geometry(spec, mujoco.mjtGeom.mjGEOM_BOX, (n / 2, sign * scene.lane_width / 2, 0.008),
                     [n / 2, 0.018, 0.004], [.45, .75, .76, 1.])
        for x, y, _, wy, bottom, _ in scene.beams:
            for sign in (-1, 1):
                geometry(spec, mujoco.mjtGeom.mjGEOM_BOX, (x, y + sign * (wy / 2 - .05), bottom / 2),
                         [0.038, 0.038, bottom / 2], [.55, .32, .20, 1.])
    else:
        for i, (x, y, wx, wy, _, z) in enumerate(scene.floors):
            for sign in (-1, 1):
                geometry(spec, mujoco.mjtGeom.mjGEOM_BOX, (x + sign * (wx / 2 - .045), y, z + 0.004),
                         [0.022, wy / 2 - .035, .005], [.58, .82, .85, 1.])
                geometry(spec, mujoco.mjtGeom.mjGEOM_BOX, (x, y + sign * (wy / 2 - .045), z + 0.004),
                         [wx / 2 - .035, .022, .005], [.58, .82, .85, 1.])
            geom = spec.geom(f"hsi_floor_{i:02d}")
            tone = (i % 3) * 0.028
            geom.rgba = [.29 + tone, .54 + tone, .60 + tone, 1.]


def decorated_model(path, scene):
    spec = mujoco.MjSpec.from_file(str(path))
    specs = spec.visual.global_
    specs.offwidth, specs.offheight, specs.fovy = WIDTH, VIEW_HEIGHT, 35
    spec.visual.headlight.ambient = [.38, .41, .46]
    spec.visual.headlight.diffuse = [.67, .68, .70]
    spec.visual.headlight.specular = [.09, .08, .07]
    spec.worldbody.add_light(pos=[-4, -5, 12], dir=[.20, .25, -1.],
                             diffuse=[.70, .66, .57], castshadow=True, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    geometry(spec, mujoco.mjtGeom.mjGEOM_PLANE, (0., 0., -0.013), [40., 40., .1],
             [.105, .155, .19, 1.])
    for v in range(-18, 19):
        saturation = .37 if v % 5 == 0 else .23
        color = [saturation * .68, saturation * .88, saturation, 1.]
        geometry(spec, mujoco.mjtGeom.mjGEOM_BOX, (v, 0., -.008), [.008, 18., .002], color)
        geometry(spec, mujoco.mjtGeom.mjGEOM_BOX, (0., v, -.008), [18., .008, .002], color)
    guides(spec, scene)
    if scene.kind == "platform_giant":
        index = len(scene.starts) // 2 + 1
    elif scene.kind == "platform_2x2":
        index = 0
    else:
        index = 0
    spawn, goal = scene.starts[index], scene.goals[index, 0]
    marker(spec, spawn, [.25, .95, .99, 1.])
    marker(spec, goal, [1., .68, .22, 1.])
    if scene.kind == "barrier":
        for other_goal in scene.goals[1:, 0]:
            marker(spec, other_goal, [1., .68, .22, 1.])
    for geom in spec.geoms:
        if geom.parent.name == "world":
            continue
        if "foot" in geom.name or "ankle" in geom.name:
            geom.rgba = [.29, .38, .47, 1.]
        elif "head" in geom.name or "arm" in geom.name:
            geom.rgba = [.67, .79, .87, 1.]
        else:
            geom.rgba = [.83, .88, .90, 1.]
    return spec.compile(), spawn, goal


def ease(t):
    return t * t * (3 - 2 * t)


def camera(scene, spawn, goal, fraction):
    cam = mujoco.MjvCamera()
    t = ease(fraction)
    if scene.kind == "platform_2x2":
        cam.lookat[:] = [0., 0., 0.35]
        cam.azimuth = 145 - 59 * t
        cam.elevation = -32 - 22 * t
        cam.distance = 8.35 - 0.25 * t
    elif scene.kind == "platform_giant":
        cam.lookat[:] = np.array([0., 0., .28]) * (1 - 0.82 * t) + (spawn + goal) / 2 * (0.82 * t)
        cam.azimuth = 133 - 30 * t
        cam.elevation = -47 + 17 * t
        cam.distance = 34.5 - 24.5 * t
    else:
        length = scene.floors[0][2]
        cam.lookat[:] = np.array([length * .49, 0., .52]) * (1 - t) + np.array([2.42, 0., .58]) * t
        cam.azimuth = 116 - 36 * t
        cam.elevation = -32 + 21 * t
        cam.distance = 17.0 - 11.4 * t
    return cam


def caption(frame, scene, config):
    im = Image.new("RGB", (WIDTH, HEIGHT), "#0b1823")
    im.paste(Image.fromarray(frame), (0, TOP))
    dr = ImageDraw.Draw(im)
    dr.rectangle((30, 20, 34, 78), fill=ACCENT)
    dr.text((49, 18), "GPC STAGE 3  /  ASTRO P2", font=font(14, True), fill=ACCENT)
    dr.text((47, 40), TITLE[scene.kind], font=font(33, True), fill=TEXT)
    dr.text((WIDTH - 31, 27), "SUPERMIMIC FSQ93", anchor="ra", font=font(16, True), fill=TEXT)
    dr.text((WIDTH - 31, 56), "MUJOCO SCENE PREVIEW", anchor="ra", font=font(13), fill=MUTED)
    if scene.kind == "platform_2x2":
        w, g = config["size_m"][0], config["gap_m"][0]
        line = f"4 platforms  /  {w:.2f} m squares  /  {g:.2f} m gaps"
        second = f"Level deck at {config['base_height_m']:.2f} m  /  goal on an adjacent platform"
    elif scene.kind == "platform_giant":
        a, b = config["gap_m"]
        line = f"{len(scene.floors)} platforms  /  {a:.2f}-{b:.2f} m gap range"
        second = f"Random deck size and elevation  /  up to {config['height_delta_m']:.2f} m step in height"
    else:
        a, b = min(g[4] for g in scene.beams), max(g[4] for g in scene.beams)
        line = f"{len(scene.beams)} beams  /  {a:.2f}-{b:.2f} m clearance in this layout"
        second = f"{scene.floors[0][2]:.0f} m corridor  /  {scene.lane_width:.2f} m valid lane between cyan lines"
    dr.text((31, HEIGHT - 69), line, font=font(21, True), fill=TEXT)
    dr.text((32, HEIGHT - 34), second, font=font(15), fill=MUTED)
    dr.ellipse((WIDTH - 271, HEIGHT - 63, WIDTH - 258, HEIGHT - 50), fill=ACCENT)
    dr.text((WIDTH - 250, HEIGHT - 67), "P2 spawn", font=font(15), fill=TEXT)
    dr.ellipse((WIDTH - 136, HEIGHT - 63, WIDTH - 123, HEIGHT - 50), fill=GOAL)
    dr.text((WIDTH - 115, HEIGHT - 67), "Goal", font=font(15), fill=TEXT)
    dr.text((WIDTH - 31, HEIGHT - 32), f"Layout seed {config['seed']}",
            anchor="ra", font=font(13), fill=MUTED)
    return im


def render(layout, output, writer, still_only):
    cfg = load_config(layout)
    scene = generate(cfg["terrain"])
    xml = scene_asset(scene, cfg["assets"]["robot_xml"], output / "mjcf" / f"{scene.kind}.xml")
    model, spawn, goal = decorated_model(xml, scene)
    states = reset_cases(cfg["assets"]["reset_cases"])[0][:1]
    state = face_goal(states, torch.as_tensor(spawn[None], dtype=torch.float32),
                      torch.as_tensor(goal[None], dtype=torch.float32))[0].numpy()
    data = mujoco.MjData(model)
    data.qpos[:7] = state[:7]
    indices = [model.jnt_qposadr[model.joint(name).id] for name in MOTOR_JOINT_NAMES]
    data.qpos[indices] = state[13:42]
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    # Still viewpoints keep both the scene footprint and P2 spawn visible.
    still = {"platform_2x2": .28, "platform_giant": .14, "barrier": .33}[scene.kind]
    frames = np.linspace(0., 1., FPS * SECONDS)
    with mujoco.Renderer(model, height=VIEW_HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera(scene, spawn, goal, still), scene_option=opt)
        caption(renderer.render(), scene, cfg["terrain"]).save(output / f"{scene.kind}.png")
        if not still_only:
            for t in frames:
                renderer.update_scene(data, camera(scene, spawn, goal, t), scene_option=opt)
                writer.append_data(np.asarray(caption(renderer.render(), scene, cfg["terrain"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--still-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.still_only:
        for layout in args.configs:
            render(layout, args.output, None, True)
        return
    with imageio.get_writer(args.output / "gpc-stage3-terrain.mp4", fps=FPS, codec="libx264",
                            quality=8, pixelformat="yuv420p",
                            output_params=["-movflags", "+faststart"]) as writer:
        for layout in args.configs:
            render(layout, args.output, writer, False)
            print(f"Rendered {layout.stem}", flush=True)


if __name__ == "__main__":
    main()
