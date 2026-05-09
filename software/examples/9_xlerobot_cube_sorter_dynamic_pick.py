import argparse
import json
import time
from collections import deque

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient


RIGHT_KEYS = [
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
]
SHOULDER_PAN_KEY = "right_arm_shoulder_pan.pos"

CORNER_ORDER = ["top_left", "top_right", "bottom_left", "bottom_right"]

HSV_RANGES = {
    "red": [
        ((0, 80, 60), (10, 255, 255)),
        ((170, 80, 60), (180, 255, 255)),
    ],
    "green": [
        ((35, 60, 50), (85, 255, 255)),
    ],
    "blue": [
        ((90, 60, 50), (130, 255, 255)),
    ],
}

DRAW_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}


def detect_colored_cubes(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    for color_name, ranges in HSV_RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            mask_total |= cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )

        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if aspect < 0.45 or aspect > 2.2:
                continue

            detections.append(
                {
                    "color": color_name,
                    "area": area,
                    "bbox": (x, y, w, h),
                    "center": (x + w // 2, y + h // 2),
                }
            )

    detections.sort(key=lambda item: item["area"], reverse=True)
    return detections


def load_pick_grid(path):
    with open(path, "r", encoding="utf-8") as f:
        grid = json.load(f)

    points = grid.get("points", {})
    missing = [name for name in CORNER_ORDER if name not in points]
    if missing:
        raise KeyError(f"Missing pick-grid points in {path}: {missing}")

    for name in points:
        for key in ("pixel", "pre_pose", "grasp_pose"):
            if key not in points[name]:
                raise KeyError(f"Missing {name}.{key} in {path}")

    grid["points"] = normalize_pick_grid_points(points)
    src = np.float32([grid["points"][name]["pixel"] for name in CORNER_ORDER])
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    homography = cv2.getPerspectiveTransform(src, dst)
    return grid, homography


def normalize_pick_grid_points(points):
    labeled_points = []
    for label, point in points.items():
        px, py = point["pixel"]
        labeled_points.append((label, float(px), float(py), point))

    top_left = min(labeled_points, key=lambda item: item[1] + item[2])
    top_right = max(labeled_points, key=lambda item: item[1] - item[2])
    bottom_left = min(labeled_points, key=lambda item: item[1] - item[2])
    bottom_right = max(labeled_points, key=lambda item: item[1] + item[2])

    mapping = {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }
    normalized = {}
    source_labels = {}
    used_source_labels = set()
    for canonical_name, (source_label, _px, _py, point) in mapping.items():
        normalized[canonical_name] = dict(point)
        normalized[canonical_name]["source_label"] = source_label
        source_labels[canonical_name] = source_label
        used_source_labels.add(source_label)

    for source_label, _px, _py, point in labeled_points:
        if source_label in used_source_labels or source_label in CORNER_ORDER:
            continue
        normalized[source_label] = dict(point)
        normalized[source_label]["source_label"] = source_label

    if any(canonical != source for canonical, source in source_labels.items()):
        print(f"[GRID] Corner labels were reordered from pixels: {source_labels}")
    else:
        print("[GRID] Corner labels match pixel order.")
    print(f"[GRID] Loaded {len(normalized)} calibration points for local interpolation.")
    return normalized


def pixel_to_uv(homography, center):
    point = np.float32([[[center[0], center[1]]]])
    mapped = cv2.perspectiveTransform(point, homography)[0][0]
    return float(mapped[0]), float(mapped[1])


def clamp01(value):
    return max(0.0, min(1.0, value))


def inside_uv_with_margin(uv, margin):
    return -margin <= uv[0] <= 1.0 + margin and -margin <= uv[1] <= 1.0 + margin


def clamp_uv(uv):
    return clamp01(uv[0]), clamp01(uv[1])


def bilinear_pose(points, pose_key, u, v):
    u = clamp01(u)
    v = clamp01(v)
    tl = points["top_left"][pose_key]
    tr = points["top_right"][pose_key]
    bl = points["bottom_left"][pose_key]
    br = points["bottom_right"][pose_key]

    pose = {}
    for key in RIGHT_KEYS:
        pose[key] = (
            (1 - u) * (1 - v) * tl[key]
            + u * (1 - v) * tr[key]
            + (1 - u) * v * bl[key]
            + u * v * br[key]
        )
    return pose


def pixel_grid_polygon(grid):
    points = grid["points"]
    return np.array(
        [
            points["top_left"]["pixel"],
            points["top_right"]["pixel"],
            points["bottom_right"]["pixel"],
            points["bottom_left"]["pixel"],
        ],
        dtype=np.float32,
    )


def inside_pixel_grid(grid, center, margin_px):
    polygon = pixel_grid_polygon(grid)
    return cv2.pointPolygonTest(polygon, (float(center[0]), float(center[1])), True) >= -margin_px


def inverse_distance_pose(points, pose_key, center, power=2.0, neighbors=4):
    weights = []
    for name, point in points.items():
        if pose_key not in point:
            continue
        px, py = point["pixel"]
        distance = max(float(np.hypot(center[0] - px, center[1] - py)), 1.0)
        weights.append((name, distance, 1.0 / (distance**power)))
    weights = sorted(weights, key=lambda item: item[1])[: max(1, neighbors)]

    total = sum(weight for _name, _distance, weight in weights)
    pose = {}
    for key in RIGHT_KEYS:
        pose[key] = sum(points[name][pose_key][key] * weight for name, _distance, weight in weights) / total
    return pose


def parse_pose_offsets(raw_offsets):
    offsets = {}
    for raw in raw_offsets:
        if "=" not in raw:
            raise ValueError(f"Offset must be JOINT=VALUE, got: {raw}")
        joint, value = raw.split("=", 1)
        joint = joint.strip()
        if joint not in RIGHT_KEYS:
            raise KeyError(f"Unknown right-arm joint for offset: {joint}")
        offsets[joint] = offsets.get(joint, 0.0) + float(value)
    return offsets


def apply_pose_offsets(pose, offsets):
    result = dict(pose)
    for joint, offset in offsets.items():
        result[joint] += offset
    return result


def blend_pose(start_pose, end_pose, alpha):
    alpha = max(0.0, min(1.2, alpha))
    return {key: start_pose[key] + (end_pose[key] - start_pose[key]) * alpha for key in RIGHT_KEYS}


def add_offset(offsets, joint, value):
    if value:
        offsets[joint] = offsets.get(joint, 0.0) + value


def pose_with_gripper(pose, gripper_pose):
    result = dict(pose)
    result["right_arm_gripper.pos"] = float(gripper_pose["right_arm_gripper.pos"])
    return result


def with_base_stop(pose):
    action = dict(pose)
    action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    return action


def get_right_pose(robot):
    obs = robot.get_observation()
    return {key: float(obs[key]) for key in RIGHT_KEYS}


def interpolate_pose(start, end, steps):
    for i in range(1, steps + 1):
        alpha = i / steps
        yield {key: start[key] + (end[key] - start[key]) * alpha for key in RIGHT_KEYS}


def move_to_pose(robot, target_pose, duration=2.0, hz=20):
    start_pose = get_right_pose(robot)
    steps = max(1, int(duration * hz))
    sleep_s = 1.0 / hz

    for pose in interpolate_pose(start_pose, target_pose, steps):
        robot.send_action(with_base_stop(pose))
        time.sleep(sleep_s)


def most_stable_target(history, stable_frames):
    if len(history) < stable_frames:
        return None, len(history)

    recent = list(history)[-stable_frames:]
    first = recent[0]
    if first and all(item and item["color"] == first["color"] for item in recent):
        avg_x = sum(item["center"][0] for item in recent) / stable_frames
        avg_y = sum(item["center"][1] for item in recent) / stable_frames
        result = dict(first)
        result["center"] = (int(avg_x), int(avg_y))
        return result, stable_frames

    last = recent[-1]
    if not last:
        return None, recent.count(None)
    return last, sum(item and item["color"] == last["color"] for item in recent)


def draw_scene(frame, detections, grid, target, uv, stable_count, stable_frames):
    output = frame.copy()
    points = grid["points"]
    polygon = np.array(
        [
            points["top_left"]["pixel"],
            points["top_right"]["pixel"],
            points["bottom_right"]["pixel"],
            points["bottom_left"]["pixel"],
        ],
        dtype=np.int32,
    )
    cv2.polylines(output, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)

    for name, point in points.items():
        px, py = point["pixel"]
        color = (255, 255, 255) if name in CORNER_ORDER else (0, 255, 255)
        cv2.circle(output, (px, py), 6, color, 2)
        cv2.putText(output, name, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    for det in detections:
        color_name = det["color"]
        x, y, w, h = det["bbox"]
        cx, cy = det["center"]
        draw_color = DRAW_COLORS[color_name]
        cv2.rectangle(output, (x, y), (x + w, y + h), draw_color, 2)
        cv2.circle(output, (cx, cy), 5, draw_color, -1)
        cv2.putText(
            output,
            f"{color_name} ({cx},{cy})",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            draw_color,
            2,
        )

    status = "no stable target"
    if target and uv:
        status = f"{target['color']} uv=({uv[0]:.2f},{uv[1]:.2f}) stable={stable_count}/{stable_frames}"
    cv2.putText(output, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, "s: dynamic sort | q/esc: quit", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def run_dynamic_pick_sort(
    robot,
    poses,
    grid,
    target,
    uv,
    pose_method,
    pre_offsets,
    grasp_offsets,
    idw_power,
    idw_neighbors,
    grasp_depth,
):
    open_gripper = poses["gripper_open"]
    closed_gripper = poses["gripper_closed"]
    target_color = target["color"]
    target_center = target["center"]
    drop_pose_name = f"drop_{target_color}"

    if pose_method == "bilinear":
        pre_pose = bilinear_pose(grid["points"], "pre_pose", uv[0], uv[1])
        grasp_pose = bilinear_pose(grid["points"], "grasp_pose", uv[0], uv[1])
    else:
        pre_pose = inverse_distance_pose(grid["points"], "pre_pose", target_center, power=idw_power, neighbors=idw_neighbors)
        grasp_pose = inverse_distance_pose(
            grid["points"], "grasp_pose", target_center, power=idw_power, neighbors=idw_neighbors
        )
    pre_pose = apply_pose_offsets(pre_pose, pre_offsets)
    grasp_pose = apply_pose_offsets(grasp_pose, grasp_offsets)
    effective_grasp_pose = blend_pose(pre_pose, grasp_pose, grasp_depth)

    sequence = [
        ("home", pose_with_gripper(poses["home"], open_gripper), 2.0),
        ("observe", pose_with_gripper(poses["observe"], open_gripper), 2.0),
        ("dynamic_pre_grasp", pose_with_gripper(pre_pose, open_gripper), 2.5),
        ("dynamic_grasp", pose_with_gripper(effective_grasp_pose, open_gripper), 2.5),
        ("close_gripper", pose_with_gripper(effective_grasp_pose, closed_gripper), 1.0),
        ("dynamic_lift", pose_with_gripper(pre_pose, closed_gripper), 2.0),
        (drop_pose_name, pose_with_gripper(poses[drop_pose_name], closed_gripper), 3.0),
        ("open_gripper", pose_with_gripper(poses[drop_pose_name], open_gripper), 1.0),
        ("home", poses["home"], 3.0),
    ]

    print(
        f"[SORT] {target_color} center={target_center} "
        f"uv=({uv[0]:.3f}, {uv[1]:.3f}) method={pose_method} grasp_depth={grasp_depth:.2f} -> {drop_pose_name}"
    )
    print(
        "[SORT] pre_pose shoulder_pan/lift/elbow="
        f"{pre_pose['right_arm_shoulder_pan.pos']:.2f}/"
        f"{pre_pose['right_arm_shoulder_lift.pos']:.2f}/"
        f"{pre_pose['right_arm_elbow_flex.pos']:.2f}"
    )
    print(
        "[SORT] grasp_pose shoulder_pan/lift/elbow/wrist_flex="
        f"{effective_grasp_pose['right_arm_shoulder_pan.pos']:.2f}/"
        f"{effective_grasp_pose['right_arm_shoulder_lift.pos']:.2f}/"
        f"{effective_grasp_pose['right_arm_elbow_flex.pos']:.2f}/"
        f"{effective_grasp_pose['right_arm_wrist_flex.pos']:.2f}"
    )
    for name, pose, duration in sequence:
        print(f"[SORT] Moving to {name}")
        move_to_pose(robot, pose, duration=duration)
    print("[SORT] Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-ip", required=True)
    parser.add_argument("--poses", default="software/examples/cube_sorter_poses.json")
    parser.add_argument("--grid", default="software/examples/cube_sorter_pick_grid.json")
    parser.add_argument("--camera", default="head_cam")
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--uv-margin", type=float, default=0.08)
    parser.add_argument("--pixel-margin", type=float, default=20.0)
    parser.add_argument("--pose-method", choices=["idw", "bilinear"], default="idw")
    parser.add_argument("--idw-neighbors", type=int, default=4)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument(
        "--grasp-depth",
        type=float,
        default=1.0,
        help="Fraction from pre-grasp to grasp pose. Use 0.85-0.95 if the gripper presses the cube.",
    )
    parser.add_argument(
        "--pre-offset",
        action="append",
        default=[],
        help="Repeatable offset for dynamic pre-grasp pose, e.g. right_arm_shoulder_lift.pos=1.5",
    )
    parser.add_argument(
        "--grasp-offset",
        action="append",
        default=[],
        help="Repeatable offset for dynamic grasp pose, e.g. right_arm_elbow_flex.pos=-1.0",
    )
    parser.add_argument(
        "--shoulder-pan-offset",
        type=float,
        default=0.0,
        help="Shortcut offset for q/e base shoulder pan. Applied to both pre-grasp and grasp poses.",
    )
    parser.add_argument(
        "--pre-shoulder-pan-offset",
        type=float,
        default=0.0,
        help="Extra q/e shoulder pan offset for pre-grasp only.",
    )
    parser.add_argument(
        "--grasp-shoulder-pan-offset",
        type=float,
        default=0.0,
        help="Extra q/e shoulder pan offset for grasp only.",
    )
    args = parser.parse_args()

    with open(args.poses, "r", encoding="utf-8") as f:
        poses = json.load(f)
    grid, homography = load_pick_grid(args.grid)
    pre_offsets = parse_pose_offsets(args.pre_offset)
    grasp_offsets = parse_pose_offsets(args.grasp_offset)
    add_offset(pre_offsets, SHOULDER_PAN_KEY, args.shoulder_pan_offset)
    add_offset(grasp_offsets, SHOULDER_PAN_KEY, args.shoulder_pan_offset)
    add_offset(pre_offsets, SHOULDER_PAN_KEY, args.pre_shoulder_pan_offset)
    add_offset(grasp_offsets, SHOULDER_PAN_KEY, args.grasp_shoulder_pan_offset)
    if pre_offsets:
        print(f"[CONFIG] pre_offsets={pre_offsets}")
    if grasp_offsets:
        print(f"[CONFIG] grasp_offsets={grasp_offsets}")

    robot = XLerobotClient(
        XLerobotClientConfig(
            remote_ip=args.pi_ip,
            id="xlerobot_cube_sorter_dynamic_pick",
            cameras={
                args.camera: OpenCVCameraConfig(
                    index_or_path=0,
                    fps=15,
                    width=640,
                    height=480,
                    color_mode=ColorMode.BGR,
                    rotation=Cv2Rotation.NO_ROTATION,
                )
            },
        )
    )
    robot.connect()
    history = deque(maxlen=max(args.stable_frames, 1))

    print("[PC] Connected.")
    print("[PC] Put one cube inside the calibrated white quadrilateral.")
    print("[PC] Press 's' to sort the stable target, q/esc to quit.")

    try:
        while True:
            obs = robot.get_observation()
            frame = obs.get(args.camera)
            if frame is None:
                print(f"[PC] No camera frame. keys={list(obs.keys())}")
                robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
                time.sleep(0.05)
                continue

            detections = detect_colored_cubes(frame, args.min_area)
            history.append(detections[0] if detections else None)
            target, stable_count = most_stable_target(history, args.stable_frames)
            uv = pixel_to_uv(homography, target["center"]) if target else None

            output = draw_scene(frame, detections, grid, target, uv, stable_count, args.stable_frames)
            cv2.imshow("XLeRobot dynamic cube sorter", output)

            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if key == ord("s"):
                if not target or stable_count < args.stable_frames or uv is None:
                    print("[PC] Target is not stable yet.")
                    continue
                if args.pose_method == "bilinear" and not inside_uv_with_margin(uv, args.uv_margin):
                    print(
                        f"[PC] Target outside calibrated grid: "
                        f"uv=({uv[0]:.3f}, {uv[1]:.3f}), margin={args.uv_margin:.2f}"
                    )
                    continue
                if args.pose_method == "idw" and not inside_pixel_grid(grid, target["center"], args.pixel_margin):
                    print(
                        f"[PC] Target outside calibrated pixel grid: "
                        f"center={target['center']}, margin_px={args.pixel_margin:.1f}"
                    )
                    continue

                cv2.destroyWindow("XLeRobot dynamic cube sorter")
                run_dynamic_pick_sort(
                    robot,
                    poses,
                    grid,
                    target,
                    clamp_uv(uv),
                    args.pose_method,
                    pre_offsets,
                    grasp_offsets,
                    args.idw_power,
                    args.idw_neighbors,
                    args.grasp_depth,
                )
                break

    finally:
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        robot.disconnect()
        cv2.destroyAllWindows()
        print("[PC] Stopped.")


if __name__ == "__main__":
    main()
