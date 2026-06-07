import numpy as np

prev_centers = {}

def detect_movement(detections):
    alerts = []

    if detections.tracker_id is None or detections.xyxy is None:
        return alerts

    for i, box in enumerate(detections.xyxy):
        if detections.class_id[i] != 0:
            continue

        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        tid = int(detections.tracker_id[i])

        if tid in prev_centers:
            px, py = prev_centers[tid]
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist > 60:
                alerts.append(f"Fast Movement — ID {tid}")

        prev_centers[tid] = (cx, cy)

    return alerts