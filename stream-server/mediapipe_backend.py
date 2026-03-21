"""
Headless MediaPipe drowsiness heuristics (from drowsyness-detection/detect.py).

Per-WebSocket instance: keeps EAR/MAR/head counters so streak logic matches the desktop demo.
No GUI, no winsound — safe for Linux / Modal.
"""
from __future__ import annotations

import base64
import math
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

# Landmark indices (same as detect.py)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_MATH = [78, 13, 308, 14]
NOSE = 1
CHIN = 152

EAR_THRESHOLD = 0.26
MAR_THRESHOLD = 0.6
HEAD_DROP_RATIO = 0.9

CLOSED_FRAMES_THRESHOLD = 10
YAWN_FRAMES_THRESHOLD = 15
LOOKING_DOWN_THRESHOLD = 20


def _aspect_ratio_eye(landmarks, points, frame_w: int, frame_h: int) -> float:
    pts = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in points]
    v1, v2 = math.dist(pts[1], pts[5]), math.dist(pts[2], pts[4])
    h = math.dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0


def _aspect_ratio_mouth(landmarks, points, frame_w: int, frame_h: int) -> float:
    pts = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in points]
    v = math.dist(pts[1], pts[3])
    h = math.dist(pts[0], pts[2])
    return v / h if h != 0 else 0.0


class MediaPipeDrowsinessTracker:
    """One instance per WebSocket client (isolated counters)."""

    def __init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._closed_count = 0
        self._yawn_count = 0
        self._down_count = 0

    def close(self) -> None:
        self._face_mesh.close()

    def analyze_base64_jpeg(self, frame_base64: str) -> Tuple[bool, float, str]:
        """
        Returns (is_drowsy, confidence, class_name) aligned with TASK_DIVISION JSON.

        is_drowsy True if eyes closed, sustained yawn, or head-down heuristic fires this frame.
        confidence: rough 0–1 strength of the active signal.
        """
        try:
            if "," in frame_base64:
                frame_base64 = frame_base64.split(",")[1]
            img_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return False, 0.0, "Error"

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                self._closed_count = self._yawn_count = self._down_count = 0
                return False, 0.5, "Non Drowsy"

            face = results.multi_face_landmarks[0]
            lms = face.landmark

            right_ear = _aspect_ratio_eye(lms, RIGHT_EYE, w, h)
            left_ear = _aspect_ratio_eye(lms, LEFT_EYE, w, h)
            avg_ear = (right_ear + left_ear) / 2.0
            mar = _aspect_ratio_mouth(lms, MOUTH_MATH, w, h)

            eye_level_y = (lms[33].y + lms[263].y) / 2.0
            upper_face_dist = abs(lms[NOSE].y - eye_level_y)
            lower_face_dist = abs(lms[CHIN].y - lms[NOSE].y)
            head_ratio = upper_face_dist / lower_face_dist if lower_face_dist != 0 else 0.0

            if avg_ear < EAR_THRESHOLD:
                self._closed_count += 1
            else:
                self._closed_count = 0

            if mar > MAR_THRESHOLD:
                self._yawn_count += 1
            else:
                self._yawn_count = 0

            if head_ratio > HEAD_DROP_RATIO:
                self._down_count += 1
            else:
                self._down_count = 0

            # Priority: eyes > head > yawn (same severity order as detect.py UI)
            if self._closed_count >= CLOSED_FRAMES_THRESHOLD:
                conf = min(1.0, 0.55 + 0.02 * min(self._closed_count, 20))
                return True, conf, "Drowsy"
            if self._down_count >= LOOKING_DOWN_THRESHOLD:
                conf = min(1.0, 0.5 + 0.015 * min(self._down_count, 20))
                return True, conf, "Distracted"
            if self._yawn_count >= YAWN_FRAMES_THRESHOLD:
                conf = min(1.0, 0.5 + 0.01 * min(self._yawn_count, 25))
                return True, conf, "Yawning"

            return False, max(avg_ear, 0.35), "Non Drowsy"

        except Exception as e:
            print(f"  MediaPipe analysis error: {e}")
            return False, 0.0, "Error"
