import time
import numpy as np

person_history = {}

def reset_history():
    global person_history
    person_history = {}

def detect_behaviour(person_id, bbox):
    x1, y1, x2, y2 = bbox
    cx  = (x1 + x2) // 2
    cy  = (y1 + y2) // 2
    now = time.time()

    if person_id not in person_history:
        person_history[person_id] = {
            "positions":   [],
            "last_label":  "Normal",
            "last_update": now,
        }

    hist = person_history[person_id]
    hist["positions"].append((cx, cy, now))

    if len(hist["positions"]) > 15:
        hist["positions"].pop(0)

    label = "Normal"

    if len(hist["positions"]) >= 4:
        speeds = []
        for i in range(1, len(hist["positions"])):
            px, py, pt = hist["positions"][i - 1]
            cx2, cy2, ct = hist["positions"][i]
            dist  = ((cx2 - px) ** 2 + (cy2 - py) ** 2) ** 0.5
            dt    = ct - pt + 1e-5
            speeds.append(dist / dt)

        avg_speed = np.mean(speeds)

        if avg_speed > 80:
            label = "Fast Move"
        elif avg_speed > 30:
            label = "Active"
        else:
            label = "Normal"

    # stability filter — no flicker
    if now - hist["last_update"] < 1.2:
        return hist["last_label"]

    hist["last_label"]  = label
    hist["last_update"] = now
    return label