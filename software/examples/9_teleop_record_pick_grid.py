import argparse
import json
import select
import sys
import termios
import time
import tty

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

GRID_POINTS = [
    ("top_left", "camera top-left reachable point"),
    ("top_center", "camera top-center reachable point"),
    ("top_right", "camera top-right reachable point"),
    ("middle_left", "camera middle-left reachable point"),
    ("center", "camera center reachable point"),
    ("middle_right", "camera middle-right reachable point"),
    ("bottom_left", "camera bottom-left reachable point"),
    ("bottom_center", "camera bottom-center reachable point"),
    ("bottom_right", "camera bottom-right reachable point"),
]
GRID_POINT_NAMES = [name for name, _description in GRID_POINTS]

HSV_RANGES = {
    "red": [
        ((0, 80, 60), (10, 255, 255)),
        ((170, 80, 60), (180, 255, 255)),
    ],
    "green": [
        ((35, 60, 50), (85, 255, 255)),
    ],
    "blue": [
        ((95, 120, 100), (130, 255, 255)),
    ],
}
DEFAULT_BLUE_MIN_SATURATION = 120
DEFAULT_BLUE_MIN_VALUE = 100

DRAW_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}


def detect_colored_cubes(
    frame,
    min_area,
    blue_min_saturation=DEFAULT_BLUE_MIN_SATURATION,
    blue_min_value=DEFAULT_BLUE_MIN_VALUE,
):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    for color_name, ranges in HSV_RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            if color_name == "blue":
                lower = (
                    lower[0],
                    max(lower[1], blue_min_saturation),
                    max(lower[2], blue_min_value),
                )
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


def draw_frame(frame, detections, point_name, stage_name, saved_points):
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
            f"{color_name} ({cx},{cy})",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            draw_color,
            2,
        )

    for saved in saved_points.values():
        pixel = saved.get("pixel")
        if pixel:
            cv2.circle(output, tuple(pixel), 7, (255, 255, 255), 2)

    cv2.putText(output, f"{point_name}: {stage_name}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, "p: save | h: help | q/esc: quit", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def read_key(timeout=0.02):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


def capture_pose(robot):
    obs = robot.get_observation()
    return {key: float(obs[key]) for key in RIGHT_KEYS}


def send_pose(robot, pose):
    action = dict(pose)
    action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    robot.send_action(action)


def print_help(point_name, description, stage_name):
    print("\n--- Pick-grid recorder ---")
    print(f"Point: {point_name} ({description})")
    print(f"Stage: {stage_name}")
    print("Place one red/green/blue cube at this grid point.")
    print("For 'pre_pose': move gripper above the cube and keep cube visible.")
    print("For 'grasp_pose': move gripper down to the cube.")
    print("Controls:")
    print("  q/e: shoulder_pan -/+")
    print("  w/s: shoulder_lift -/+")
    print("  a/d: elbow_flex -/+")
    print("  r/f: wrist_flex -/+")
    print("  z/x: wrist_roll -/+")
    print("  t/g: gripper -/+")
    print("  p: save current stage")
    print("  h: print help")
    print("  b or esc in terminal/window: quit")
    print("--------------------------\n")


def build_keymap(joint_step, gripper_step):
    return {
        "q": ("right_arm_shoulder_pan.pos", -joint_step),
        "e": ("right_arm_shoulder_pan.pos", joint_step),
        "w": ("right_arm_shoulder_lift.pos", -joint_step),
        "s": ("right_arm_shoulder_lift.pos", joint_step),
        "a": ("right_arm_elbow_flex.pos", -joint_step),
        "d": ("right_arm_elbow_flex.pos", joint_step),
        "r": ("right_arm_wrist_flex.pos", -joint_step),
        "f": ("right_arm_wrist_flex.pos", joint_step),
        "z": ("right_arm_wrist_roll.pos", -joint_step),
        "x": ("right_arm_wrist_roll.pos", joint_step),
        "t": ("right_arm_gripper.pos", -gripper_step),
        "g": ("right_arm_gripper.pos", gripper_step),
    }


def load_existing_grid(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("points", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-ip", required=True)
    parser.add_argument("--camera", default="head_cam")
    parser.add_argument("--in-file", default=None)
    parser.add_argument("--out", default="software/examples/cube_sorter_pick_grid.json")
    parser.add_argument("--points", nargs="+", choices=GRID_POINT_NAMES, default=None)
    parser.add_argument("--joint-step", type=float, default=2.0)
    parser.add_argument("--gripper-step", type=float, default=2.0)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--blue-min-saturation", type=int, default=DEFAULT_BLUE_MIN_SATURATION)
    parser.add_argument("--blue-min-value", type=int, default=DEFAULT_BLUE_MIN_VALUE)
    args = parser.parse_args()

    keymap = build_keymap(args.joint_step, args.gripper_step)
    robot = XLerobotClient(
        XLerobotClientConfig(
            remote_ip=args.pi_ip,
            id="pick_grid_recorder",
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

    current_pose = capture_pose(robot)
    saved_points = load_existing_grid(args.in_file)
    points_to_record = [(name, description) for name, description in GRID_POINTS if args.points is None or name in args.points]
    if args.in_file:
        print(f"Loaded existing grid from {args.in_file}: {len(saved_points)} points")
    if args.points:
        print(f"Updating only points: {', '.join(args.points)}")
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    try:
        for point_name, description in points_to_record:
            saved_points[point_name] = dict(saved_points.get(point_name, {}))

            for stage_name in ("pre_pose", "grasp_pose"):
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                print_help(point_name, description, stage_name)
                tty.setraw(sys.stdin.fileno())

                while True:
                    obs = robot.get_observation()
                    frame = obs.get(args.camera)
                    detections = (
                        detect_colored_cubes(
                            frame,
                            args.min_area,
                            args.blue_min_saturation,
                            args.blue_min_value,
                        )
                        if frame is not None
                        else []
                    )
                    output = draw_frame(frame, detections, point_name, stage_name, saved_points) if frame is not None else None
                    if output is not None:
                        cv2.imshow("XLeRobot pick-grid recorder", output)

                    send_pose(robot, current_pose)

                    window_key = cv2.waitKey(1) & 0xFF
                    key = read_key()
                    if window_key in (ord("b"), 27) or key in ("b", "\x1b", "\x03"):
                        raise KeyboardInterrupt

                    if key == "h":
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        print_help(point_name, description, stage_name)
                        tty.setraw(sys.stdin.fileno())
                        continue

                    if key == "p":
                        if stage_name == "pre_pose":
                            if not detections:
                                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                                print("\nNo colored cube detected. Put a visible red/green/blue cube at this point first.")
                                tty.setraw(sys.stdin.fileno())
                                continue
                            saved_points[point_name]["pixel"] = list(map(int, detections[0]["center"]))
                            saved_points[point_name]["detected_color"] = detections[0]["color"]

                        saved_points[point_name][stage_name] = dict(current_pose)
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        print(f"\nSaved {point_name}.{stage_name}")
                        print(json.dumps(saved_points[point_name], indent=2))
                        tty.setraw(sys.stdin.fileno())
                        break

                    if key in keymap:
                        joint, delta = keymap[key]
                        current_pose[joint] += delta
                        send_pose(robot, current_pose)

                    time.sleep(0.01)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        payload = {
            "version": 1,
            "camera": args.camera,
            "points": saved_points,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved pick-grid calibration to {args.out}")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
