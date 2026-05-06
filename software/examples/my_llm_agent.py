print("1. Script started", flush=True)

from dotenv import load_dotenv
load_dotenv()

from robocrew.core.camera import RobotCamera
from robocrew.core.LLMAgent import LLMAgent

from robocrew.robots.XLeRobot.tools import (
    create_move_forward,
    create_move_backward,
    create_turn_right,
    create_turn_left,
    create_strafe_right,
    create_strafe_left,
    create_look_around,
    create_go_to_precision_mode,
    create_go_to_normal_mode,
)

from robocrew.robots.XLeRobot.servo_controls import ServoControler

print("2. Imports loaded", flush=True)

main_camera = RobotCamera("/dev/video0")
print("3. Camera loaded", flush=True)

right_arm_wheel_usb = "/dev/arm_right"
left_arm_head_usb = "/dev/arm_left"

servo_controler = ServoControler(
    right_arm_wheel_usb,
    left_arm_head_usb
)
print("4. Servo controller loaded", flush=True)

move_forward = create_move_forward(servo_controler)
move_backward = create_move_backward(servo_controler)
turn_left = create_turn_left(servo_controler)
turn_right = create_turn_right(servo_controler)
strafe_left = create_strafe_left(servo_controler)
strafe_right = create_strafe_right(servo_controler)
look_around = create_look_around(servo_controler, main_camera)
go_to_precision_mode = create_go_to_precision_mode(servo_controler)
go_to_normal_mode = create_go_to_normal_mode(servo_controler)

print("5. Movement tools created", flush=True)

agent = LLMAgent(
    model="google_genai:gemini-3-flash-preview",
    tools=[
        move_forward,
        move_backward,
        strafe_left,
        strafe_right,
        turn_left,
        turn_right,
        look_around,
        go_to_precision_mode,
        go_to_normal_mode,
    ],
    main_camera=main_camera,
    servo_controler=servo_controler,
    history_len=8,
)

print("6. Agent created", flush=True)

agent.task = (
    "Follow the human using the camera. "
    "If the human is visible and far away, move_forward by 0.2 meters. "
    "If the human is on the left, turn_left. "
    "If the human is on the right, turn_right. "
    "If the human is very close, stop. "
    "Do not use arms. Do not grab anything."
)

print("7. Starting agent", flush=True)

agent.go()
