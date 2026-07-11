import cv2
import mediapipe as mp
import numpy as np
import time
import threading

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# --- MATH HELPERS ---
def calculate_angle(a, b, c):
    """Calculate the angle at point b formed by rays b->a and b->c, in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def get_xy(landmarks, landmark):
    lm = landmarks[landmark.value]
    return [lm.x, lm.y]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# --- MODULAR TEST CLASSES ---
class FitnessTest:
    """Base class for all fitness/combine tests.

    Tests are "armed" on creation but do not run until start() is called
    explicitly (from the Start button). Every test reports a 0-100
    `quality_score`: a composite of joint angles and normalized movement
    distance, not just a rep tally. Raw rep count is still tracked (self.reps)
    but is secondary information, not the headline number.
    """

    MIN_VISIBILITY = 0.5

    def __init__(self):
        self.reps = 0
        self.feedback = "Press Start when you're ready"
        self.is_done = False
        self.form_warnings = 0
        self.started = False
        self.rep_scores = []       # history of per-rep composite scores (0-100)
        self.quality_score = 0     # rolling average of rep_scores, the headline metric

    def start(self):
        self.started = True
        self.feedback = "Go!"

    def process_frame(self, landmarks, frame_shape):
        if not self.started or self.is_done:
            return self.quality_score, self.feedback
        return self._run(landmarks, frame_shape)

    def _run(self, landmarks, frame_shape):
        raise NotImplementedError("Each test must implement _run")

    def _record_rep(self, score):
        score = clamp(score, 0, 100)
        self.rep_scores.append(score)
        self.reps += 1
        self.quality_score = round(sum(self.rep_scores) / len(self.rep_scores))

    def snapshot_extra(self):
        """Optional extra fields merged into the live_stats payload."""
        return {}

    def _avg_visibility(self, landmarks, indices):
        vis = [landmarks[i].visibility for i in indices]
        return sum(vis) / len(vis) if vis else 0.0


class HighKneeTest(FitnessTest):
    """10s high-knee / running-on-the-spot test.

    Scoring is built from three things measured on every rep:
      - Lift distance: how close the knee gets to hip height, normalized by
        the person's own thigh length (hip-to-knee distance measured while
        the leg is extended), so it works at any distance from the camera.
      - Elbow angle: how close the arm pump sits to an efficient ~90 degrees.
      - Torso lean: how upright the back stays.
    Reps are still counted, but the headline number is the composite
    quality_score (0-100), averaged across every rep performed.
    """

    KNEE_UP_ANGLE = 110      # hip-knee-ankle angle below this = knee driven up (rep starts)
    KNEE_DOWN_ANGLE = 160    # above this = leg extended again (rep cycle closes, score is finalized)
    STANDING_ANGLE = 150     # above this we treat the leg as "resting" and recalibrate thigh length
    TARGET_LIFT_RATIO = 0.9  # knee reaching ~90% of the way to hip height scores full marks
    ELBOW_TARGET = 90.0
    ELBOW_TOLERANCE = 60.0
    TORSO_LEAN_LIMIT = 30.0
    THIGH_REF_ALPHA = 0.15   # EMA smoothing for the self-calibrated thigh-length reference

    REQUIRED_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value,
    ]

    def __init__(self, duration=10.0):
        super().__init__()
        self.duration = duration
        self.start_time = None
        self.last_angles = {}
        self.live_lift_pct = 0.0
        self.side = {
            "LEFT": self._new_side_state(),
            "RIGHT": self._new_side_state(),
        }

    @staticmethod
    def _new_side_state():
        return {"stage": "down", "thigh_ref": None, "peak_ratio": 0.0, "peak_elbow": 180.0, "peak_lean": 0.0}

    def start(self):
        super().start()
        self.start_time = time.time()

    def _run(self, landmarks, frame_shape):
        elapsed = time.time() - self.start_time
        remaining = max(0.0, self.duration - elapsed)

        if self._avg_visibility(landmarks, self.REQUIRED_LANDMARKS) < self.MIN_VISIBILITY:
            self.feedback = "Step back so your full body is visible"
            return self.quality_score, self.feedback

        L = mp_pose.PoseLandmark
        pts = {}
        for s in ("LEFT", "RIGHT"):
            pts[s] = {
                "shoulder": get_xy(landmarks, getattr(L, f"{s}_SHOULDER")),
                "hip": get_xy(landmarks, getattr(L, f"{s}_HIP")),
                "knee": get_xy(landmarks, getattr(L, f"{s}_KNEE")),
                "ankle": get_xy(landmarks, getattr(L, f"{s}_ANKLE")),
                "elbow": get_xy(landmarks, getattr(L, f"{s}_ELBOW")),
                "wrist": get_xy(landmarks, getattr(L, f"{s}_WRIST")),
            }

        knee_angles, elbow_angles = {}, {}
        for s in ("LEFT", "RIGHT"):
            knee_angles[s] = calculate_angle(pts[s]["hip"], pts[s]["knee"], pts[s]["ankle"])
            elbow_angles[s] = calculate_angle(pts[s]["shoulder"], pts[s]["elbow"], pts[s]["wrist"])

        self.last_angles = {
            "left_knee": round(knee_angles["LEFT"], 1),
            "right_knee": round(knee_angles["RIGHT"], 1),
            "left_elbow": round(elbow_angles["LEFT"], 1),
            "right_elbow": round(elbow_angles["RIGHT"], 1),
        }

        # Posture: torso lean relative to vertical
        mid_shoulder = np.mean([pts["LEFT"]["shoulder"], pts["RIGHT"]["shoulder"]], axis=0)
        mid_hip = np.mean([pts["LEFT"]["hip"], pts["RIGHT"]["hip"]], axis=0)
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.5]
        torso_lean = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

        live_ratio = 0.0
        for s in ("LEFT", "RIGHT"):
            st = self.side[s]
            hip, knee = pts[s]["hip"], pts[s]["knee"]

            # Continuously (re)calibrate this leg's resting thigh length while it's extended.
            if knee_angles[s] > self.STANDING_ANGLE:
                thigh_len = float(np.linalg.norm(np.array(hip) - np.array(knee)))
                st["thigh_ref"] = thigh_len if st["thigh_ref"] is None else (
                    (1 - self.THIGH_REF_ALPHA) * st["thigh_ref"] + self.THIGH_REF_ALPHA * thigh_len
                )

            # Lift ratio: 0 = knee at rest height, 1.0 = knee level with the hip.
            if st["thigh_ref"] and st["thigh_ref"] > 1e-6:
                gap = knee[1] - hip[1]  # normally positive (knee below hip in image coords)
                ratio = clamp(1.0 - gap / st["thigh_ref"], 0.0, 1.4)
            else:
                ratio = 0.0
            live_ratio = max(live_ratio, ratio)

            if st["stage"] == "down" and knee_angles[s] < self.KNEE_UP_ANGLE:
                st["stage"] = "up"
                st["peak_ratio"] = ratio
                st["peak_elbow"] = elbow_angles[s]
                st["peak_lean"] = torso_lean
            elif st["stage"] == "up":
                if ratio > st["peak_ratio"]:
                    st["peak_ratio"] = ratio
                    st["peak_elbow"] = elbow_angles[s]
                    st["peak_lean"] = torso_lean
                if knee_angles[s] > self.KNEE_DOWN_ANGLE:
                    st["stage"] = "down"
                    self._score_rep(st["peak_ratio"], st["peak_elbow"], st["peak_lean"])

        self.live_lift_pct = live_ratio * 100

        # Live, frame-by-frame coaching feedback (independent of per-rep scoring above)
        form_msgs = []
        if torso_lean > self.TORSO_LEAN_LIMIT:
            form_msgs.append("Chest up, keep your back straight")
        if live_ratio < 0.5:
            form_msgs.append("Drive those knees higher")
        arm_ok = all(abs(elbow_angles[s] - self.ELBOW_TARGET) <= self.ELBOW_TOLERANCE for s in ("LEFT", "RIGHT"))
        if not arm_ok:
            form_msgs.append("Pump your arms at ~90 degrees")

        if form_msgs:
            self.form_warnings += 1
            self.feedback = " | ".join(form_msgs)
        else:
            self.feedback = f"Great form! {remaining:.1f}s left"

        if remaining <= 0 and not self.is_done:
            self.is_done = True
            self.feedback = f"Done! Form score: {self.quality_score}/100 over {self.reps} reps"

        return self.quality_score, self.feedback

    def _score_rep(self, peak_ratio, peak_elbow, peak_lean):
        lift_component = 50 * clamp(peak_ratio / self.TARGET_LIFT_RATIO, 0, 1)
        elbow_component = 30 * max(0.0, 1 - abs(peak_elbow - self.ELBOW_TARGET) / self.ELBOW_TOLERANCE)
        posture_component = 20 * max(0.0, 1 - peak_lean / self.TORSO_LEAN_LIMIT)
        self._record_rep(lift_component + elbow_component + posture_component)

    def snapshot_extra(self):
        elapsed = 0.0 if self.start_time is None else time.time() - self.start_time
        return {
            "angles": self.last_angles,
            "reps": self.reps,
            "lift_pct": round(self.live_lift_pct, 1),
            "time_remaining": round(max(0.0, self.duration - elapsed), 1),
            "form_warnings": self.form_warnings,
        }


class SquatTest(FitnessTest):
    """Bodyweight squat test, scored on depth distance, L/R symmetry, and lockout.

    Depth is measured as how far the hips drop below their standing height,
    normalized by the person's own standing leg length (hip-to-ankle
    distance), rather than relying on knee angle alone.
    """

    DEPTH_ANGLE = 100        # avg knee angle considered "in the hole"
    LOCKOUT_ANGLE = 160      # avg knee angle considered fully stood up
    TARGET_DEPTH_RATIO = 0.5  # hips dropping ~50% of leg length scores full marks
    SYMMETRY_TOLERANCE = 40.0

    REQUIRED_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    def __init__(self):
        super().__init__()
        self.stage = "up"
        self.last_angles = {}
        self.leg_ref = None
        self.baseline_hip_y = None
        self.peak_depth_ratio = 0.0
        self.peak_asymmetry = 0.0
        self.live_depth_pct = 0.0

    def _run(self, landmarks, frame_shape):
        if self._avg_visibility(landmarks, self.REQUIRED_LANDMARKS) < self.MIN_VISIBILITY:
            self.feedback = "Step back so your full body is visible"
            return self.quality_score, self.feedback

        L = mp_pose.PoseLandmark

        def pt(name, side):
            return get_xy(landmarks, getattr(L, f"{side}_{name}"))

        l_hip, r_hip = pt("HIP", "LEFT"), pt("HIP", "RIGHT")
        l_knee, r_knee = pt("KNEE", "LEFT"), pt("KNEE", "RIGHT")
        l_ankle, r_ankle = pt("ANKLE", "LEFT"), pt("ANKLE", "RIGHT")

        left_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_angle = calculate_angle(r_hip, r_knee, r_ankle)
        avg_angle = (left_angle + right_angle) / 2
        asymmetry = abs(left_angle - right_angle)

        self.last_angles = {"left_knee": round(left_angle, 1), "right_knee": round(right_angle, 1)}

        avg_hip_y = (l_hip[1] + r_hip[1]) / 2
        avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2
        leg_len = abs(avg_ankle_y - avg_hip_y)

        # Calibrate standing reference (leg length + hip height) while standing tall
        if avg_angle > self.LOCKOUT_ANGLE - 5:
            self.leg_ref = leg_len if self.leg_ref is None else (0.85 * self.leg_ref + 0.15 * leg_len)
            self.baseline_hip_y = avg_hip_y if self.baseline_hip_y is None else (0.85 * self.baseline_hip_y + 0.15 * avg_hip_y)

        depth_ratio = 0.0
        if self.leg_ref and self.baseline_hip_y is not None and self.leg_ref > 1e-6:
            depth_ratio = clamp((avg_hip_y - self.baseline_hip_y) / self.leg_ref, 0.0, 1.2)
        self.live_depth_pct = depth_ratio * 100

        if avg_angle < self.DEPTH_ANGLE:
            self.stage = "down"
            self.peak_depth_ratio = max(self.peak_depth_ratio, depth_ratio)
            self.peak_asymmetry = max(self.peak_asymmetry, asymmetry)
            if asymmetry > self.SYMMETRY_TOLERANCE:
                self.feedback = "Even out your weight, one knee is leading"
            else:
                self.feedback = "Good depth — now drive up"
        elif avg_angle > self.LOCKOUT_ANGLE and self.stage == "down":
            self.stage = "up"
            self._score_rep(self.peak_depth_ratio, self.peak_asymmetry)
            self.feedback = "Lockout! Nice rep."
            self.peak_depth_ratio = 0.0
            self.peak_asymmetry = 0.0
        elif self.stage == "up":
            self.feedback = "Break at the hips, squat down"

        return self.quality_score, self.feedback

    def _score_rep(self, peak_depth_ratio, peak_asymmetry):
        depth_component = 60 * clamp(peak_depth_ratio / self.TARGET_DEPTH_RATIO, 0, 1)
        symmetry_component = 40 * max(0.0, 1 - peak_asymmetry / self.SYMMETRY_TOLERANCE)
        self._record_rep(depth_component + symmetry_component)

    def snapshot_extra(self):
        return {
            "angles": self.last_angles,
            "reps": self.reps,
            "depth_pct": round(self.live_depth_pct, 1),
        }


class JumpTest(FitnessTest):
    """Vertical jump test, scored on jump height distance.

    Uses a short standing calibration window to learn the person's standing
    hip-to-ankle leg length in the camera's normalized coordinates, then uses
    that as a real-world ruler (assuming an average adult leg length) to
    convert vertical foot displacement into approximate centimeters.
    quality_score maps the best jump against a target height, so it reads on
    the same 0-100 scale as the other two tests.
    """

    CALIBRATION_TIME = 2.0
    LEG_REFERENCE_CM = 90.0
    GROUND_TOLERANCE = 0.02
    MIN_JUMP_CM = 5.0
    TARGET_JUMP_CM = 60.0   # a jump at/above this height scores full marks

    REQUIRED_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    def __init__(self, duration=15.0):
        super().__init__()
        self.duration = duration
        self.start_time = None
        self.calibrated = False
        self.baseline_ankle_y = None
        self.baseline_leg_len = None
        self.in_air = False
        self.current_jump_peak_cm = 0.0
        self.best_jump_cm = 0.0
        self.last_jump_cm = 0.0

    def start(self):
        super().start()
        self.start_time = time.time()

    def _run(self, landmarks, frame_shape):
        elapsed = time.time() - self.start_time
        remaining = max(0.0, self.duration - elapsed)

        if self._avg_visibility(landmarks, self.REQUIRED_LANDMARKS) < self.MIN_VISIBILITY:
            self.feedback = "Step back so your hips and feet are both visible"
            return self.quality_score, self.feedback

        L = mp_pose.PoseLandmark
        l_hip, r_hip = get_xy(landmarks, L.LEFT_HIP), get_xy(landmarks, L.RIGHT_HIP)
        l_ankle, r_ankle = get_xy(landmarks, L.LEFT_ANKLE), get_xy(landmarks, L.RIGHT_ANKLE)

        avg_hip_y = (l_hip[1] + r_hip[1]) / 2
        avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2
        leg_len = abs(avg_ankle_y - avg_hip_y)

        if elapsed < self.CALIBRATION_TIME:
            self.feedback = f"Stand still — calibrating ({self.CALIBRATION_TIME - elapsed:.1f}s)"
            self.baseline_ankle_y = avg_ankle_y
            self.baseline_leg_len = leg_len
            return self.quality_score, self.feedback

        if not self.calibrated:
            self.calibrated = True
            self.feedback = "Calibrated — jump!"

        px_to_cm = (self.LEG_REFERENCE_CM / self.baseline_leg_len) if self.baseline_leg_len else 0.0
        displacement = self.baseline_ankle_y - avg_ankle_y
        displacement_cm = max(0.0, displacement * px_to_cm)

        if displacement > self.GROUND_TOLERANCE:
            self.in_air = True
            self.current_jump_peak_cm = max(self.current_jump_peak_cm, displacement_cm)
            self.feedback = f"Airborne — {displacement_cm:.0f} cm"
        else:
            if self.in_air and self.current_jump_peak_cm >= self.MIN_JUMP_CM:
                self.last_jump_cm = self.current_jump_peak_cm
                self.best_jump_cm = max(self.best_jump_cm, self.last_jump_cm)
                self._record_rep(100 * clamp(self.last_jump_cm / self.TARGET_JUMP_CM, 0, 1))
                self.feedback = f"Landed! {self.last_jump_cm:.0f} cm — best {self.best_jump_cm:.0f} cm"
            elif self.calibrated:
                self.feedback = "Bend your knees and jump"
            self.in_air = False
            self.current_jump_peak_cm = 0.0

        if remaining <= 0 and not self.is_done:
            self.is_done = True
            self.feedback = f"Done! Best jump: {self.best_jump_cm:.0f} cm — score {self.quality_score}/100"

        return self.quality_score, self.feedback

    def snapshot_extra(self):
        elapsed = 0.0 if self.start_time is None else time.time() - self.start_time
        return {
            "reps": self.reps,
            "best_jump_cm": round(self.best_jump_cm, 1),
            "last_jump_cm": round(self.last_jump_cm, 1),
            "calibrating": elapsed < self.CALIBRATION_TIME,
            "time_remaining": round(max(0.0, self.duration - elapsed), 1),
        }


TEST_REGISTRY = {
    "high_knees": lambda: HighKneeTest(duration=10.0),
    "squat": lambda: SquatTest(),
    "jump": lambda: JumpTest(duration=15.0),
}


# --- LIVE STATE MANAGER ---
class LiveState:
    def __init__(self):
        self._lock = threading.Lock()
        self.is_streaming = False
        self.active_test = None
        self.test_name = "none"
        self.current_score = 0
        self.current_feedback = "Waiting to start..."
        self.current_extra = {}
        self.is_done = False

    def set_test(self, test_name):
        with self._lock:
            self.test_name = test_name
            self.current_score = 0
            self.current_feedback = "Press Start when you're ready"
            self.current_extra = {}
            self.is_done = False
            factory = TEST_REGISTRY.get(test_name)
            self.active_test = factory() if factory else None

    def start_test(self):
        """Explicitly begins the currently-armed test (Start button)."""
        with self._lock:
            if self.active_test:
                self.active_test.start()
                self.current_feedback = self.active_test.feedback

    def restart_test(self):
        """Resets the current test back to its unstarted state (Restart button)."""
        with self._lock:
            name = self.test_name
        self.set_test(name)

    def snapshot(self):
        with self._lock:
            payload = {
                "test": self.test_name,
                "score": self.current_score,
                "feedback": self.current_feedback,
                "done": self.is_done,
                "started": bool(self.active_test.started) if self.active_test else False,
            }
            payload.update(self.current_extra)
            return payload


live_state = LiveState()


def _draw_hud(frame, score, feedback, warn):
    color = (70, 200, 255) if not warn else (60, 90, 255)  # BGR: amber-ish vs red-ish
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (10, 10, 12), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    cv2.putText(frame, f"SCORE: {score}", (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, feedback[:60], (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def generate_live_stream():
    live_state.is_streaming = True
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened() and live_state.is_streaming:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror so it feels natural to the user
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            with live_state._lock:
                test = live_state.active_test

            warn = False
            if results.pose_landmarks and test and not test.is_done:
                score, feedback = test.process_frame(results.pose_landmarks.landmark, frame.shape)
                warn = ("|" in feedback) or ("Step back" in feedback) or ("still" in feedback.lower())

                with live_state._lock:
                    live_state.current_score = score
                    live_state.current_feedback = feedback
                    live_state.current_extra = test.snapshot_extra()
                    live_state.is_done = test.is_done

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            elif not results.pose_landmarks:
                with live_state._lock:
                    live_state.current_feedback = "No person detected — step into frame"

            with live_state._lock:
                disp_score = live_state.current_score
                disp_msg = live_state.current_feedback

            _draw_hud(frame, disp_score, disp_msg, warn)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        pose.close()
        live_state.is_streaming = False