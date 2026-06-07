import numpy as np

# track how long each ID has been in frame
id_first_seen = {}

def detect_suspicious(detections):
    alerts = []

    if detections.tracker_id is None or detections.xyxy is None:
        return alerts

    import time
    now = time.time()

    for i, tid in enumerate(detections.tracker_id):
        if detections.class_id[i] != 0:
            continue

        # loitering — in frame for too long
        if tid not in id_first_seen:
            id_first_seen[tid] = now
        elif now - id_first_seen[tid] > 30:
            alerts.append(f"Loitering — ID {tid}")

    # zone boundary (left/right edge)
    if detections.xyxy is not None:
        frame_w = 640
        for i, box in enumerate(detections.xyxy):
            if detections.class_id[i] != 0:
                continue
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            if cx < frame_w * 0.05 or cx > frame_w * 0.95:
                tid = detections.tracker_id[i] if detections.tracker_id is not None else "?"
                alerts.append(f"Boundary Alert — ID {tid}")

    return alerts