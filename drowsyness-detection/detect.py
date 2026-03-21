import cv2
import mediapipe as mp
import math
import numpy as np
import winsound
import threading
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95] 
MOUTH_MATH = [78, 13, 308, 14] 
NOSE = 1
CHIN = 152

LOW_POLY_OUTLINE = [10, 332, 454, 365, 152, 136, 234, 103]

# --- THRESHOLDS ---
EAR_THRESHOLD = 0.26
MAR_THRESHOLD = 0.6  
HEAD_DROP_RATIO = 0.9 

# --- ALARM BUFFERS ---
CLOSED_FRAMES_THRESHOLD = 10
YAWN_FRAMES_THRESHOLD = 15
LOOKING_DOWN_THRESHOLD = 20

closed_count = 0
yawn_count = 0
down_count = 0

# --- AUDIO ALARM SETUP ---
last_alarm_time = 0
ALARM_COOLDOWN = 1.0  # Wait 1 second before playing the next beep

def play_beep(freq, duration):
    """Runs in a separate thread to prevent video lag"""
    winsound.Beep(freq, duration)

def trigger_alarm(alert_type):
    global last_alarm_time
    current_time = time.time()
    
    # Check if the cooldown has passed
    if current_time - last_alarm_time > ALARM_COOLDOWN:
        if alert_type == "CRITICAL":
            # High pitch, long beep
            threading.Thread(target=play_beep, args=(2500, 500), daemon=True).start()
        elif alert_type == "WARNING":
            # Mid pitch, short beep
            threading.Thread(target=play_beep, args=(1500, 300), daemon=True).start()
        elif alert_type == "STATUS":
            # Low pitch, short bloop
            threading.Thread(target=play_beep, args=(800, 200), daemon=True).start()
            
        last_alarm_time = current_time

def get_aspect_ratio(landmarks, points, frame_w, frame_h, is_mouth=False):
    pts = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in points]
    if not is_mouth:
        v1, v2 = math.dist(pts[1], pts[5]), math.dist(pts[2], pts[4])
        h = math.dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h != 0 else 0
    else:
        v = math.dist(pts[1], pts[3]) 
        h = math.dist(pts[0], pts[2]) 
        return v / h if h != 0 else 0

def get_coords(landmarks, indices, w, h):
    return np.array([[(int(landmarks[i].x * w), int(landmarks[i].y * h))] for i in indices], dtype=np.int32)

cap = cv2.VideoCapture(0)
print("Audio-Enabled Low-Poly UI loaded... Hit 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    state = "SYSTEMS: OK" 
    main_color = (0, 255, 0) # Green

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lms = face_landmarks.landmark
            
            # --- MATH ---
            right_ear = get_aspect_ratio(lms, RIGHT_EYE, w, h)
            left_ear = get_aspect_ratio(lms, LEFT_EYE, w, h)
            avg_ear = (right_ear + left_ear) / 2.0

            mar = get_aspect_ratio(lms, MOUTH_MATH, w, h, is_mouth=True)

            eye_level_y = (lms[33].y + lms[263].y) / 2.0 
            upper_face_dist = abs(lms[NOSE].y - eye_level_y)
            lower_face_dist = abs(lms[CHIN].y - lms[NOSE].y)
            head_ratio = upper_face_dist / lower_face_dist if lower_face_dist != 0 else 0

            # --- LOGIC & COUNTERS ---
            if avg_ear < EAR_THRESHOLD: closed_count += 1
            else: closed_count = 0

            if mar > MAR_THRESHOLD: yawn_count += 1
            else: yawn_count = 0

            if head_ratio > HEAD_DROP_RATIO: down_count += 1
            else: down_count = 0

            # --- STATE MACHINE & AUDIO TRIGGERS ---
            if closed_count >= CLOSED_FRAMES_THRESHOLD:
                state = "ERROR: WAKE UP!"
                main_color = (0, 0, 255) # Red
                trigger_alarm("CRITICAL")
                
            elif down_count >= LOOKING_DOWN_THRESHOLD:
                state = "WARNING: DISTRACTED!"
                main_color = (0, 165, 255) # Orange
                trigger_alarm("WARNING")
                
            elif yawn_count >= YAWN_FRAMES_THRESHOLD:
                state = "STATUS: YAWNING"
                main_color = (0, 255, 255) # Yellow
                trigger_alarm("STATUS")

            # --- VISUAL DRAWING ---
            
            # 1. Draw Geometric Eye Contours
            eye_color = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
            cv2.polylines(frame, [get_coords(lms, RIGHT_EYE, w, h)], isClosed=True, color=eye_color, thickness=1)
            cv2.polylines(frame, [get_coords(lms, LEFT_EYE, w, h)], isClosed=True, color=eye_color, thickness=1)

            # 2. Draw Geometric Mouth Contour
            mouth_color = (0, 255, 255) if mar > MAR_THRESHOLD else (0, 255, 0)
            cv2.polylines(frame, [get_coords(lms, MOUTH_OUTLINE, w, h)], isClosed=True, color=mouth_color, thickness=1)

            # 3. Draw The Jagged Low-Poly Outline
            poly_pts = get_coords(lms, LOW_POLY_OUTLINE, w, h)
            cv2.polylines(frame, [poly_pts], isClosed=True, color=main_color, thickness=2, lineType=cv2.LINE_AA)
            
            # --- Dynamic State Text (Floats above head) ---
            forehead_x = int(lms[10].x * w)
            forehead_y = int(lms[10].y * h)
            
            text_size = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = forehead_x - (text_size[0] // 2)
            text_y = max(forehead_y - 30, 30) 
            
            cv2.putText(frame, state, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, main_color, 2, cv2.LINE_AA)

            # --- DEBUG UI (Top Left) ---
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Head: {head_ratio:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Advanced Driver Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()