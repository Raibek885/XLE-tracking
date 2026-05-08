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


def draw_detections(frame, detections, stable_color, stable_count, stable_frames):
    output = frame.copy()

    for det in detections:
        color_name = det["color"]
        x, y, w, h = det["bbox"]
        cx, cy = det["center"]
        draw_color = DRAW_COLORS[color_name]

        cv2.rectangle(output, (x, y), (x + w, y + h), draw_color, 2)
        cv2.circle(output, (cx, cy), 5, draw_color, -1)
        cv2.putText(
            output,
            f"{color_name} ({cx},{cy}) area={int(det['area'])}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            draw_color,
            2,
        )

    status = "no stable target"
    if stable_color:
        status = f"stable={stable_color} {stable_count}/{stable_frames}"
    cv2.putText(output, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(output, "s: sort stable target | q/esc: quit", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def with_base_stop(pose):
    action = dict(pose)
    action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    return action


def get_right_pose(robot):
    obs = robot.get_observation()
    return {key: float(obs[key]) for key in RIGHT_KEYS}


def pose_with_gripper(pose, gripper_pose):
    result = dict(pose)
    result["right_arm_gripper.pos"] = float(gripper_pose["right_arm_gripper.pos"])
    return result


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


def run_fixed_pick_sort(robot, poses, target_color):
    open_gripper = poses["gripper_open"]
    closed_gripper = poses["gripper_closed"]
    drop_pose_name = f"drop_{target_color}"

    sequence = [
        ("home", pose_with_gripper(poses["home"], open_gripper), 2.0),
        ("observe", pose_with_gripper(poses["observe"], open_gripper), 2.0),
        ("pre_grasp_table_center", pose_with_gripper(poses["pre_grasp_table_center"], open_gripper), 2.5),
        ("grasp_table_center", pose_with_gripper(poses["grasp_table_center"], open_gripper), 2.5),
        ("close_gripper", pose_with_gripper(poses["grasp_table_center"], closed_gripper), 1.0),
        ("lift_after_grasp", pose_with_gripper(poses["lift_after_grasp"], closed_gripper), 2.0),
        (drop_pose_name, pose_with_gripper(poses[drop_pose_name], closed_gripper), 3.0),
        ("open_gripper", pose_with_gripper(poses[drop_pose_name], open_gripper), 1.0),
        ("home", pose_with_gripper(poses["home"], open_gripper), 3.0),
    ]

    print(f"[SORT] Detected {target_color}. Running fixed pick -> {drop_pose_name}")
    for name, pose, duration in sequence:
        print(f"[SORT] Moving to {name}")
        move_to_pose(robot, pose, duration=duration)
    print("[SORT] Done.")


def most_stable_color(history, stable_frames):
    if len(history) < stable_frames:
        return None, len(history)
    recent = list(history)[-stable_frames:]
    first = recent[0]
    if first is not None and all(color == first for color in recent):
        return first, stable_frames
    if recent[-1] is None:
        return None, recent.count(None)
    return recent[-1], sum(color == recent[-1] for color in recent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-ip", required=True)
    parser.add_argument("--poses", default="software/examples/cube_sorter_poses.json")
    parser.add_argument("--camera", default="head_cam")
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    with open(args.poses, "r", encoding="utf-8") as f:
        poses = json.load(f)

    required_poses = {
        "home",
        "observe",
        "pre_grasp_table_center",
        "grasp_table_center",
        "lift_after_grasp",
        "drop_red",
        "drop_green",
        "drop_blue",
        "gripper_open",
        "gripper_closed",
    }
    missing = sorted(required_poses.difference(poses))
    if missing:
        raise KeyError(f"Missing poses in {args.poses}: {missing}")

    robot = XLerobotClient(
        XLerobotClientConfig(
            remote_ip=args.pi_ip,
            id="xlerobot_cube_sorter_fixed_pick",
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
    print("[PC] Put one cube at the recorded grasp_table_center point.")
    print("[PC] Press 's' to sort the stable detected color, q/esc to quit.")
    if args.auto:
        print("[PC] AUTO mode enabled: sorting starts as soon as the color is stable.")

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
            top_color = detections[0]["color"] if detections else None
            history.append(top_color)
            stable_color, stable_count = most_stable_color(history, args.stable_frames)

            output = draw_detections(frame, detections, stable_color, stable_count, args.stable_frames)
            cv2.imshow("XLeRobot fixed-pick cube sorter", output)

            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})

            key = cv2.waitKey(1) & 0xFF
            should_sort = stable_color is not None and stable_count >= args.stable_frames and (args.auto or key == ord("s"))

            if key == ord("q") or key == 27:
                break
            if should_sort:
                cv2.destroyWindow("XLeRobot fixed-pick cube sorter")
                run_fixed_pick_sort(robot, poses, stable_color)
                break

    finally:
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        robot.disconnect()
        cv2.destroyAllWindows()
        print("[PC] Stopped.")


if __name__ == "__main__":
    main()
