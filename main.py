import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading

# -------------------------------
# Initialize MediaPipe Face Mesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize drawing utilities.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# -------------------------------
# Setup pyttsx3 for voice alerts
# -------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speech rate if needed.

# -------------------------------
# Constants and Global Flags
# -------------------------------
# Landmark indices for eyes.
LEFT_EYE_INDICES  = [33, 133, 159, 145, 153, 154]
RIGHT_EYE_INDICES = [362, 263, 386, 374, 380, 381]

# Warning system parameters:
warning_duration     = 5.0    # Sliding window duration (seconds) for frequency counting.
frequency_threshold  = 40     # Level 1 events within 'warning_duration' to trigger warning.
continuous_threshold = 0.8    # Continuous time (seconds) in Level 1 to trigger warning.
revocation_level3    = 3      # Continuous Level 3 (Fully Healthy) duration (seconds) to revoke warning.
revocation_level2    = 8      # Continuous Level 2 (Anxious/Angry) duration (seconds) to revoke warning.

# Global state flags and time trackers.
sleep_warning = False
sleepy_event_timestamps = []   # Timestamps for Level 1 events.
continuous_level1_start = None  # When Level 1 first started continuously.
continuous_level2_start = None  # When Level 2 first started continuously.
continuous_level3_start = None  # When Level 3 first started continuously.

# --------------------------------------
# Voice Alert Thread – Continuous Speech
# --------------------------------------
def voice_alert():
    global sleep_warning
    while True:
        # As long as sleep_warning is active, repeatedly say the warning.
        if sleep_warning:
            engine.say("Warning drowsy driver")
            engine.runAndWait()
            # A brief pause between announcements.
            time.sleep(0.5)
        else:
            time.sleep(0.1)

# Start the voice alert thread (daemon=True ensures it closes when main thread ends)
voice_thread = threading.Thread(target=voice_alert, daemon=True)
voice_thread.start()

# --------------------------------------
# Helper Function: Calculate Eye Aspect Ratio (EAR)
# --------------------------------------
def calculate_eye_aspect_ratio(landmarks, indices, image_width, image_height):
    """Convert normalized landmarks to pixel coordinates and compute the EAR."""
    def to_pixel(landmark):
        return np.array([int(landmark.x * image_width), int(landmark.y * image_height)])
    
    p1 = to_pixel(landmarks[indices[0]])
    p2 = to_pixel(landmarks[indices[1]])
    p3 = to_pixel(landmarks[indices[2]])
    p4 = to_pixel(landmarks[indices[3]])
    p5 = to_pixel(landmarks[indices[4]])
    p6 = to_pixel(landmarks[indices[5]])
    
    horizontal_dist = np.linalg.norm(p1 - p2)
    vertical_dist1  = np.linalg.norm(p3 - p4)
    vertical_dist2  = np.linalg.norm(p5 - p6)
    
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist) if horizontal_dist != 0 else 0
    return ear

# --------------------------------------
# Main Loop: Capture video and analyze frames
# --------------------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for a selfie-view.
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the frame from BGR to RGB for processing.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Initialize driver state and EAR.
    driver_state = "Unknown"
    display_ear = 0.0
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh on the frame.
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                drawing_spec,
                drawing_spec
            )

            # Compute EAR for both eyes.
            left_ear = calculate_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_INDICES, frame_width, frame_height)
            right_ear = calculate_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_INDICES, frame_width, frame_height)
            display_ear = (left_ear + right_ear) / 2.0

            # Determine driver's state based on EAR thresholds.
            if display_ear < 0.18:
                driver_state = "Level 1 - Sleepy / Very Drowsy"
            elif display_ear < 0.22:
                driver_state = "Level 2 - Anxious / Angry"
            else:
                driver_state = "Level 3 - Fully Healthy / Normal"

            # Display the state and EAR.
            cv2.putText(frame, f"State: {driver_state}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {display_ear:.2f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- Sleep Warning and Revocation Logic ---
            # If driver is in Level 1 (sleepy state)
            if driver_state.startswith("Level 1"):
                if continuous_level1_start is None:
                    continuous_level1_start = current_time
                sleepy_event_timestamps.append(current_time)
                # Reset the healthy state timers.
                continuous_level2_start = None
                continuous_level3_start = None
            else:
                continuous_level1_start = None

            # For Level 2 state (Anxious/Angry)
            if driver_state.startswith("Level 2"):
                if continuous_level2_start is None:
                    continuous_level2_start = current_time
                continuous_level3_start = None
            else:
                continuous_level2_start = None

            # For Level 3 state (Fully Healthy/Normal)
            if driver_state.startswith("Level 3"):
                if continuous_level3_start is None:
                    continuous_level3_start = current_time
                continuous_level2_start = None
            else:
                continuous_level3_start = None

            # Keep only Level 1 events within the sliding window.
            sleepy_event_timestamps = [t for t in sleepy_event_timestamps if current_time - t <= warning_duration]

            # Activate Sleep Warning if:
            #   (a) Level 1 state persists continuously longer than 'continuous_threshold' or
            #   (b) Frequency threshold within the sliding window is met.
            if driver_state.startswith("Level 1"):
                if (continuous_level1_start is not None and (current_time - continuous_level1_start) > continuous_threshold) or \
                   (len(sleepy_event_timestamps) >= frequency_threshold):
                    sleep_warning = True

            # Revocation conditions:
            # If driver is in Level 2 (Anxious/Angry) continuously for revocation_level2 seconds, revoke warning.
            if driver_state.startswith("Level 2") and continuous_level2_start is not None:
                if (current_time - continuous_level2_start) >= revocation_level2:
                    sleep_warning = False
                    sleepy_event_timestamps.clear()
            # If driver is in Level 3 (Fully Healthy) continuously for revocation_level3 seconds, revoke warning.
            if driver_state.startswith("Level 3") and continuous_level3_start is not None:
                if (current_time - continuous_level3_start) >= revocation_level3:
                    sleep_warning = False
                    sleepy_event_timestamps.clear()

    # If sleep warning is active, overlay a red warning label.
    if sleep_warning:
        cv2.putText(frame, "SLEEPY DRIVER!", (50, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit if the 'Esc' key is pressed.
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()