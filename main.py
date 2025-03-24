import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh with refined landmarks.
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

# Landmark indices for eyes.
# (These indices are typical with MediaPipe Face Mesh. Adjust if needed.)
LEFT_EYE_INDICES  = [33, 133, 159, 145, 153, 154]
RIGHT_EYE_INDICES = [362, 263, 386, 374, 380, 381]

# Parameters for the warning system:
warning_duration    = 5.0    # Sliding window duration (seconds) for frequency counting.
frequency_threshold = 40     # Level 1 events within 5 seconds to trigger the warning.
continuous_threshold = 0.8    # Continuous Level 1 time (in sec) to trigger the warning.
revocation_level3    = 4     # Continuous healthy (Level 3) duration to revoke warning (sec).
revocation_level2    = 8     # Continuous anxious/angry (Level 2) duration to revoke warning (sec).

# Persistent variables across frames.
sleep_warning = False
sleepy_event_timestamps = []  # List to store timestamps of Level 1 events.
continuous_level1_start = None  # Time when Level 1 was first detected continuously.
continuous_level2_start = None  # Time when Level 2 was first detected continuously.
continuous_level3_start = None  # Time when Level 3 was first detected continuously.

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for a selfie-view and get dimensions.
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert from BGR to RGB for MediaPipe.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    driver_state = "Unknown"
    display_ear = 0.0
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the full face mesh.
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
            
            # Determine driver's state based on EAR.
            if display_ear < 0.18:
                driver_state = "Level 1 - Sleepy / Very Drowsy"
            elif display_ear < 0.22:
                driver_state = "Level 2 - Anxious / Angry"
            else:
                driver_state = "Level 3 - Fully Healthy / Normal"
            
            # Display state and EAR.
            cv2.putText(
                frame, f"State: {driver_state}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"EAR: {display_ear:.2f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # ----- Sleep Warning & Revocation Logic -----
            # If in Level 1, update the Level 1 timer and frequency events.
            if driver_state.startswith("Level 1"):
                if continuous_level1_start is None:
                    continuous_level1_start = current_time
                # Record this Level 1 detection.
                sleepy_event_timestamps.append(current_time)
                # Reset revocation timers.
                continuous_level2_start = None
                continuous_level3_start = None
            else:
                # Not in Level 1 → reset level 1 timer.
                continuous_level1_start = None

            # For Level 2 state:
            if driver_state.startswith("Level 2"):
                if continuous_level2_start is None:
                    continuous_level2_start = current_time
                # Reset other revocation timer.
                continuous_level3_start = None
            else:
                continuous_level2_start = None

            # For Level 3 state:
            if driver_state.startswith("Level 3"):
                if continuous_level3_start is None:
                    continuous_level3_start = current_time
                continuous_level2_start = None
            else:
                continuous_level3_start = None

            # Clean up old level 1 events (only keep events within the sliding window).
            sleepy_event_timestamps = [t for t in sleepy_event_timestamps if current_time - t <= warning_duration]

            # If currently in Level 1, check if either condition is met:
            if driver_state.startswith("Level 1"):
                # Check if continuously in Level 1 longer than threshold or frequency threshold reached.
                if (continuous_level1_start is not None and (current_time - continuous_level1_start) > continuous_threshold) or \
                   (len(sleepy_event_timestamps) >= frequency_threshold):
                    sleep_warning = True

            # Revocation conditions:
            # If driver is in Level 2 for a continuous period longer than revocation_level2 seconds, clear the warning.
            if driver_state.startswith("Level 2") and continuous_level2_start is not None:
                if (current_time - continuous_level2_start) >= revocation_level2:
                    sleep_warning = False
                    sleepy_event_timestamps.clear()
            # If driver is in Level 3 for a continuous period longer than revocation_level3 seconds, clear the warning.
            if driver_state.startswith("Level 3") and continuous_level3_start is not None:
                if (current_time - continuous_level3_start) >= revocation_level3:
                    sleep_warning = False
                    sleepy_event_timestamps.clear()
    
    # Display the sleepy driver warning (in red) if active.
    if sleep_warning:
        cv2.putText(
            frame, "SLEEPY DRIVER!", (50, frame_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4
        )
    
    cv2.imshow("Driver Drowsiness Detection", frame)
    
    # Exit if 'Esc' key is pressed.
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()