import cv2
import mediapipe as mp
import numpy as np
import time
import threading

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- MATH HELPERS ---
def calculate_angle(a, b, c):
    """Calculate angle between 3 points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- MODULAR TEST CLASSES ---
class FitnessTest:
    """Base class for all fitness tests."""
    def __init__(self):
        self.score = 0
        self.feedback = "Ready!"
        self.is_done = False

    def process_frame(self, landmarks, frame_shape):
        raise NotImplementedError("Each test must implement process_frame")

class HighKneeTest(FitnessTest):
    """Measures running on the spot (high knees) & posture."""
    def __init__(self, duration=10.0):
        super().__init__()
        self.duration = duration
        self.start_time = None
        self.reps = 0
        self.stage = "down"
        self.posture_warnings = 0

    def process_frame(self, landmarks, frame_shape):
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.is_done = True
            return self.reps, f"Done! Score: {self.reps} reps. Posture flags: {self.posture_warnings}"

        L = mp_pose.PoseLandmark
        
        # Get coordinates
        def get_pt(landmark):
            return [landmarks[landmark.value].x, landmarks[landmark.value].y]

        l_hip, r_hip = get_pt(L.LEFT_HIP), get_pt(L.RIGHT_HIP)
        l_knee, r_knee = get_pt(L.LEFT_KNEE), get_pt(L.RIGHT_KNEE)
        l_shoulder, r_shoulder = get_pt(L.LEFT_SHOULDER), get_pt(L.RIGHT_SHOULDER)

        # 1. Posture Check (Back Straightness)
        mid_shoulder = [(l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2]
        mid_hip = [(l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2]
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.5]
        
        torso_lean = calculate_angle(mid_shoulder, mid_hip, vertical_ref)
        is_leaning = torso_lean > 15.0 # degrees

        if is_leaning:
            self.posture_warnings += 1
            self.feedback = "Keep your back straight! Chest up!"
        else:
            self.feedback = f"Driving! Time: {max(0, self.duration - elapsed):.1f}s"

        # 2. High Knee Check
        # In MediaPipe, smaller Y is higher on the screen.
        # A high knee rep counts when a knee gets close to or above the hip line.
        avg_hip_y = (l_hip[1] + r_hip[1]) / 2
        highest_knee_y = min(l_knee[1], r_knee[1])

        if highest_knee_y < avg_hip_y + 0.05: # Knee is up
            if self.stage == "down":
                self.reps += 1
                self.stage = "up"
        elif highest_knee_y > avg_hip_y + 0.15: # Both knees dropped
            self.stage = "down"

        return self.reps, self.feedback


class SquatTest(FitnessTest):
    """Simplified version of your squat logic for modularity."""
    def __init__(self):
        super().__init__()
        self.reps = 0
        self.stage = "up"

    def process_frame(self, landmarks, frame_shape):
        L = mp_pose.PoseLandmark
        def get_pt(landmark):
            return [landmarks[landmark.value].x, landmarks[landmark.value].y, landmarks[landmark.value].z]

        # Calculate one side for brevity in this example
        hip = get_pt(L.RIGHT_HIP)
        knee = get_pt(L.RIGHT_KNEE)
        ankle = get_pt(L.RIGHT_ANKLE)
        
        knee_angle = calculate_angle(hip, knee, ankle)

        if knee_angle < 90:
            self.stage = "down"
            self.feedback = "Good depth, drive up!"
        elif knee_angle > 160 and self.stage == "down":
            self.stage = "up"
            self.reps += 1
            self.feedback = "Lockout! Nice rep."
        
        return self.reps, self.feedback

# --- LIVE STATE MANAGER ---
class LiveState:
    def __init__(self):
        self._lock = threading.Lock()
        self.is_streaming = False
        self.active_test = None
        self.test_name = "None"
        self.current_score = 0
        self.current_feedback = "Waiting to start..."

    def set_test(self, test_name):
        with self._lock:
            self.test_name = test_name
            if test_name == "high_knees":
                self.active_test = HighKneeTest(duration=10.0)
            elif test_name == "squat":
                self.active_test = SquatTest()
            else:
                self.active_test = None

    def snapshot(self):
        with self._lock:
            return {
                "test": self.test_name,
                "score": self.current_score,
                "feedback": self.current_feedback
            }

live_state = LiveState()

def generate_live_stream():
    live_state.is_streaming = True
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened() and live_state.is_streaming:
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            with live_state._lock:
                test = live_state.active_test

            if results.pose_landmarks and test and not test.is_done:
                score, feedback = test.process_frame(results.pose_landmarks.landmark, frame.shape)
                
                with live_state._lock:
                    live_state.current_score = score
                    live_state.current_feedback = feedback

                # Draw skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw HUD
            with live_state._lock:
                disp_score = live_state.current_score
                disp_msg = live_state.current_feedback
                
            cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"SCORE: {disp_score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, disp_msg, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        pose.close()