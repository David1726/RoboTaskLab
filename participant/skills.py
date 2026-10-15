"""EDIT BLOCK 1: named skill execution. Use only context.observe() and context.step()."""

import math


def execute(call, context):
    if call["name"] in {"probe_cartesian", "insert", "assemble_nut", "stack"}:
        return manufacturing_skill(call, context)
    if call["name"] != "pick_place":
        raise ValueError("unknown skill")
    offset = float(call["arguments"].get("grasp_offset_m", 0.03))
    if not math.isfinite(offset) or not -0.01 <= offset <= 0.15:
        raise ValueError("invalid grasp offset")
    obs = context.observe()
    target = call["arguments"]["target"]
    destination = call["arguments"]["destination"]
    initial = obs["objects"][target][:]
    basket = obs["objects"][destination][:]

    def move(goal, grip, maximum=75, minimum=5):
        observation = context.observe()
        for tick in range(maximum):
            error = [g - p for g, p in zip(goal, observation["eef_pos"], strict=True)]
            action = [max(-0.6, min(0.6, value * 12)) for value in error] + [0, 0, 0, grip]
            observation = context.step(action)
            if tick >= minimum and math.dist(goal, observation["eef_pos"]) < 0.008:
                break
        return observation

    approach = initial[:2] + [0.27]
    grasp = initial[:2] + [initial[2] + offset]
    move(approach, -1)
    move(grasp, -1)
    closed = move(grasp, 1, maximum=30, minimum=29)
    clearance = closed["eef_pos"][2] - closed["objects"][target][2]
    xy_error = math.dist(closed["eef_pos"][:2], closed["objects"][target][:2])
    lifted = move(approach, 1)
    rise = lifted["objects"][target][2] - initial[2]
    if rise < 0.045:
        return {
            "status": "empty_grasp",
            "close_vertical_clearance_m": clearance,
            "close_xy_error_m": xy_error,
            "object_lift_m": rise,
        }
    transport = basket[:2] + [0.27]
    release = basket[:2] + [0.16]
    move(transport, 1, maximum=100)
    move(release, 1)
    move(release, -1, maximum=35, minimum=34)
    move(transport, -1)
    move(transport, -1, maximum=35, minimum=34)
    return {"status": "completed", "object_lift_m": rise}


def manufacturing_skill(call, context):
    """Frozen state-feedback primitives; all poses are achieved by robot actions."""
    arguments = call["arguments"]
    scale = float(arguments.get("motion_scale", 1.0))
    if not math.isfinite(scale) or not 0.4 <= scale <= 2.5:
        raise ValueError("invalid motion scale")
    grasp_offset = float(arguments.get("grasp_offset_m", 0.015))
    if not math.isfinite(grasp_offset) or not -0.005 <= grasp_offset <= 0.04:
        raise ValueError("invalid manufacturing grasp offset")

    def move(goal, grip, maximum=110, minimum=8, align=False):
        observation = context.observe()
        for tick in range(maximum):
            error = [g - p for g, p in zip(goal, observation["eef_pos"], strict=True)]
            action = [max(-0.65, min(0.65, v * 10 * scale)) for v in error] + [0, 0, 0, grip]
            if align:
                w, x, y, z = observation["object_quat_wxyz"]["peg"]
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                action[5] = max(-0.4, min(0.4, -yaw * 3))
            observation = context.step(action)
            if tick >= minimum and math.dist(goal, observation["eef_pos"]) < 0.004:
                break
        return observation

    obs = context.observe()
    if call["name"] == "probe_cartesian":
        move(obs["eef_pos"][:2] + [1.08], -1, maximum=70)
        start = context.observe()["eef_pos"][:]
        for _ in range(20):
            obs = context.step([0.12, 0, 0, 0, 0, 0, -1])
        end = obs["eef_pos"][:]
        response = (end[0] - start[0]) / (20 * 0.12 * 0.05)
        for _ in range(10):
            context.step([0, 0, 0, 0, 0, 0, -1])
        return {
            "status": "measured",
            "response_gain": response,
            "before_eef_m": start,
            "after_eef_m": end,
            "command_x_sum": 2.4,
        }

    source = {"insert": "peg", "assemble_nut": "nut", "stack": "block"}[call["name"]]
    destination = {"insert": "hole", "assemble_nut": "shaft", "stack": "base"}[call["name"]]
    initial = obs["objects"][source][:]
    grasp = obs.get("handle_pos", initial)[:] if source == "nut" else initial[:]
    grasp[2] += grasp_offset
    safe = 1.10 if source == "nut" else 1.04
    move(grasp[:2] + [safe], -1)
    move(grasp, -1)
    move(grasp, 1, maximum=30, minimum=29)
    lifted = move(grasp[:2] + [safe], 1)
    rise = lifted["objects"][source][2] - initial[2]
    if rise < 0.045:
        return {"status": "empty_grasp", "object_lift_m": rise}
    offset = [a - b for a, b in zip(lifted["eef_pos"], lifted["objects"][source], strict=True)]
    destination_pos = lifted["objects"][destination]
    xy = [destination_pos[i] + offset[i] for i in range(2)]
    move(xy + [safe], 1, align=source == "peg")
    if source == "peg":
        # Plate top is 16 mm above its body center; peg half-length is 50 mm.
        z = destination_pos[2] + 0.016 + 0.05 - 0.026 + offset[2]
    elif source == "nut":
        z = obs["table_z"] + 0.025 + offset[2]
    else:
        z = destination_pos[2] + 0.025 + 0.020 + 0.002 + offset[2]
    lowered = move(xy + [z], 1, maximum=130, align=source == "peg")
    lateral = math.dist(lowered["objects"][source][:2], destination_pos[:2])
    move(xy + [z], -1, maximum=30, minimum=29)
    move(xy + [safe], -1)
    for _ in range(40):
        context.step([0, 0, 0, 0, 0, 0, -1])
    return {
        "status": "completed",
        "object_lift_m": rise,
        "lateral_error_m": lateral,
        "motion_scale": scale,
    }
