import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# ── CONNECTIONS to draw ───────────────────────────────────────────────────────
POSE_CONNECTIONS = solutions.pose.POSE_CONNECTIONS

BASE_OPTIONS = mp.tasks.BaseOptions(
    model_asset_path="pose_landmarker.task"
)

OPTIONS = vision.PoseLandmarkerOptions(
    base_options=BASE_OPTIONS,
    output_segmentation_masks=False,
    num_poses=5,
    min_pose_detection_confidence=0.4,
    min_pose_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)

landmarker = vision.PoseLandmarker.create_from_options(OPTIONS)


def draw_skeleton(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    if x2 - x1 < 20 or y2 - y1 < 20:
        return frame

    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)

    try:
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return frame

        for pose_landmarks in result.pose_landmarks:
            # build NormalizedLandmarkList for drawing
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            landmark_list.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=lm.x, y=lm.y, z=lm.z
                ) for lm in pose_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                roi,
                landmark_list,
                POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

        frame[y1:y2, x1:x2] = roi

    except Exception as e:
        pass

    return frame