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
    """Helper to extract [x, y] coordinates from MediaPipe landmarks."""
    lm = landmarks[landmark.value]
    return [lm.x, lm.y]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# --- MODULAR TEST CLASSES ---
class FitnessTest:
    """Base class for all fitness/combine tests."""
    MIN_VISIBILITY = 0.5

    def __init__(self):
        self.reps = 0
        self.feedback = "System ready. Awaiting manual start."
        self.is_done = False
        self.form_warnings = 0
        self.started = False
        self.rep_scores = []       
        self.quality_score = 0     

    def start(self):
        self.started = True
        self.feedback = "Test initiated."

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
        return {}

    def _avg_visibility(self, landmarks, indices):
        vis = [landmarks[i].visibility for i in indices]
        return sum(vis) / len(vis) if vis else 0.0


class RunningInPlaceTest(FitnessTest):
    """Running-on-the-spot tracking optimal ground turnover and sagittal plane biomechanics."""
    STEP_UP_ANGLE = 155      
    STEP_DOWN_ANGLE = 170    
    IDEAL_LEAN = 8.0         
    LEAN_TOLERANCE = 18.0
    ELBOW_TARGET = 90.0
    ELBOW_TOLERANCE = 45.0
    CADENCE_TARGET = 170.0   
    CADENCE_TOLERANCE = 55.0
    CADENCE_WINDOW = 6       

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
        self.step_timestamps = []
        self.live_cadence = 0.0
        self.side = {"LEFT": self._new_side_state(), "RIGHT": self._new_side_state()}

    @staticmethod
    def _new_side_state():
        return {"stage": "down", "peak_elbow": 180.0, "peak_lean": 0.0}

    def start(self):
        super().start()
        self.start_time = time.time()

    def _run(self, landmarks, frame_shape):
        elapsed = time.time() - self.start_time
        remaining = max(0.0, self.duration - elapsed)

        if self._avg_visibility(landmarks, self.REQUIRED_LANDMARKS) < self.MIN_VISIBILITY:
            self.feedback = "Step back so your full anatomical structure is visible"
            return self.quality_score, self.feedback

        L = mp_pose.PoseLandmark
        pts = {
            s: {
                "shoulder": get_xy(landmarks, getattr(L, f"{s}_SHOULDER")),
                "hip": get_xy(landmarks, getattr(L, f"{s}_HIP")),
                "knee": get_xy(landmarks, getattr(L, f"{s}_KNEE")),
                "ankle": get_xy(landmarks, getattr(L, f"{s}_ANKLE")),
                "elbow": get_xy(landmarks, getattr(L, f"{s}_ELBOW")),
                "wrist": get_xy(landmarks, getattr(L, f"{s}_WRIST")),
            } for s in ("LEFT", "RIGHT")
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

        mid_shoulder = np.mean([pts["LEFT"]["shoulder"], pts["RIGHT"]["shoulder"]], axis=0)
        mid_hip = np.mean([pts["LEFT"]["hip"], pts["RIGHT"]["hip"]], axis=0)
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.5]
        torso_lean = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

        now = time.time()
        for s in ("LEFT", "RIGHT"):
            st = self.side[s]
            if st["stage"] == "down" and knee_angles[s] < self.STEP_UP_ANGLE:
                st["stage"] = "up"
                st["peak_elbow"] = elbow_angles[s]
                st["peak_lean"] = torso_lean
                self.step_timestamps.append(now)
                if len(self.step_timestamps) > self.CADENCE_WINDOW:
                    self.step_timestamps.pop(0)
            elif st["stage"] == "up":
                if abs(elbow_angles[s] - self.ELBOW_TARGET) > abs(st["peak_elbow"] - self.ELBOW_TARGET):
                    st["peak_elbow"] = elbow_angles[s]
                st["peak_lean"] = max(st["peak_lean"], torso_lean)
                if knee_angles[s] > self.STEP_DOWN_ANGLE:
                    st["stage"] = "down"
                    self._score_step(st["peak_elbow"], st["peak_lean"])

        if len(self.step_timestamps) >= 2:
            intervals = [t2 - t1 for t1, t2 in zip(self.step_timestamps, self.step_timestamps[1:])]
            avg_interval = sum(intervals) / len(intervals)
            self.live_cadence = 60.0 / avg_interval if avg_interval > 0 else 0.0

        form_msgs = []
        if abs(torso_lean - self.IDEAL_LEAN) > self.LEAN_TOLERANCE:
            form_msgs.append("Optimize torso alignment (target 8° forward lean)")
        if not all(abs(elbow_angles[s] - self.ELBOW_TARGET) <= self.ELBOW_TOLERANCE for s in ("LEFT", "RIGHT")):
            form_msgs.append("Maintain strict 90° sagittal arm swing")
        if self.live_cadence and self.live_cadence < self.CADENCE_TARGET - self.CADENCE_TOLERANCE:
            form_msgs.append("Increase turnover cadence (target 170+ SPM)")

        if form_msgs:
            self.form_warnings += 1
            self.feedback = " | ".join(form_msgs)
        else:
            self.feedback = f"Optimal stride mechanics. {remaining:.1f}s"

        if remaining <= 0 and not self.is_done:
            self.is_done = True
            self.feedback = f"Done! Form score: {self.quality_score}/100 over {self.reps} steps"

        return self.quality_score, self.feedback

    def _score_step(self, peak_elbow, peak_lean):
        cadence_component = (
            35 * clamp(1 - abs(self.live_cadence - self.CADENCE_TARGET) / self.CADENCE_TOLERANCE, 0, 1)
            if self.live_cadence else 0.0
        )
        posture_component = 35 * clamp(1 - abs(peak_lean - self.IDEAL_LEAN) / self.LEAN_TOLERANCE, 0, 1)
        arm_component = 30 * clamp(1 - abs(peak_elbow - self.ELBOW_TARGET) / self.ELBOW_TOLERANCE, 0, 1)
        self._record_rep(cadence_component + posture_component + arm_component)

    def snapshot_extra(self):
        elapsed = 0.0 if self.start_time is None else time.time() - self.start_time
        return {
            "angles": self.last_angles,
            "reps": self.reps,
            "cadence_spm": round(self.live_cadence, 0),
            "cadence_pct": round(clamp(self.live_cadence / self.CADENCE_TARGET * 100, 0, 140), 1),
            "time_remaining": round(max(0.0, self.duration - elapsed), 1),
            "form_warnings": self.form_warnings,
        }


class HighKneeTest(FitnessTest):
    """10s high-knee test evaluating knee lift distance, hip flexion, and lumbar posture."""
    KNEE_UP_ANGLE = 110      
    KNEE_DOWN_ANGLE = 160    
    STANDING_ANGLE = 150     
    TARGET_LIFT_RATIO = 0.9  
    ELBOW_TARGET = 90.0
    ELBOW_TOLERANCE = 60.0
    TORSO_LEAN_LIMIT = 30.0
    THIGH_REF_ALPHA = 0.15   

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
            self.feedback = "Step back so your full anatomical structure is visible"
            return self.quality_score, self.feedback

        L = mp_pose.PoseLandmark
        pts = {
            s: {
                "shoulder": get_xy(landmarks, getattr(L, f"{s}_SHOULDER")),
                "hip": get_xy(landmarks, getattr(L, f"{s}_HIP")),
                "knee": get_xy(landmarks, getattr(L, f"{s}_KNEE")),
                "ankle": get_xy(landmarks, getattr(L, f"{s}_ANKLE")),
                "elbow": get_xy(landmarks, getattr(L, f"{s}_ELBOW")),
                "wrist": get_xy(landmarks, getattr(L, f"{s}_WRIST")),
            } for s in ("LEFT", "RIGHT")
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

        mid_shoulder = np.mean([pts["LEFT"]["shoulder"], pts["RIGHT"]["shoulder"]], axis=0)
        mid_hip = np.mean([pts["LEFT"]["hip"], pts["RIGHT"]["hip"]], axis=0)
        vertical_ref = [mid_hip[0], mid_hip[1] - 0.5]
        torso_lean = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

        live_ratio = 0.0
        for s in ("LEFT", "RIGHT"):
            st = self.side[s]
            hip, knee = pts[s]["hip"], pts[s]["knee"]

            if knee_angles[s] > self.STANDING_ANGLE:
                thigh_len = float(np.linalg.norm(np.array(hip) - np.array(knee)))
                st["thigh_ref"] = thigh_len if st["thigh_ref"] is None else (
                    (1 - self.THIGH_REF_ALPHA) * st["thigh_ref"] + self.THIGH_REF_ALPHA * thigh_len
                )

            if st["thigh_ref"] and st["thigh_ref"] > 1e-6:
                gap = knee[1] - hip[1]
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

        form_msgs = []
        if torso_lean > self.TORSO_LEAN_LIMIT:
            form_msgs.append("Maintain vertical torso; avoid lumbar flexion")
        if live_ratio < 0.5:
            form_msgs.append("Incomplete hip flexion (drive knees to parallel)")
        
        arm_ok = all(abs(elbow_angles[s] - self.ELBOW_TARGET) <= self.ELBOW_TOLERANCE for s in ("LEFT", "RIGHT"))
        if not arm_ok:
            form_msgs.append("Engage contralateral arm drive")

        if form_msgs:
            self.form_warnings += 1
            self.feedback = " | ".join(form_msgs)
        else:
            self.feedback = f"Excellent triple-flexion coordination. {remaining:.1f}s"

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


class JumpTest(FitnessTest):
    """Vertical countermovement jump test."""
    CALIBRATION_TIME = 2.0
    LEG_REFERENCE_CM = 90.0
    GROUND_TOLERANCE = 0.02
    MIN_JUMP_CM = 5.0
    TARGET_JUMP_CM = 60.0   

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
            self.feedback = f"Establishing anatomical baseline... ({self.CALIBRATION_TIME - elapsed:.1f}s)"
            self.baseline_ankle_y = avg_ankle_y
            self.baseline_leg_len = leg_len
            return self.quality_score, self.feedback

        if not self.calibrated:
            self.calibrated = True
            self.feedback = "Baseline set. Execute maximum countermovement jump!"

        px_to_cm = (self.LEG_REFERENCE_CM / self.baseline_leg_len) if self.baseline_leg_len else 0.0
        displacement = self.baseline_ankle_y - avg_ankle_y
        displacement_cm = max(0.0, displacement * px_to_cm)

        if displacement > self.GROUND_TOLERANCE:
            self.in_air = True
            self.current_jump_peak_cm = max(self.current_jump_peak_cm, displacement_cm)
            self.feedback = f"Flight phase... {displacement_cm:.0f} cm"
        else:
            if self.in_air and self.current_jump_peak_cm >= self.MIN_JUMP_CM:
                self.last_jump_cm = self.current_jump_peak_cm
                self.best_jump_cm = max(self.best_jump_cm, self.last_jump_cm)
                self._record_rep(100 * clamp(self.last_jump_cm / self.TARGET_JUMP_CM, 0, 1))
                self.feedback = f"Landing registered: {self.last_jump_cm:.0f} cm (Peak: {self.best_jump_cm:.0f} cm)"
            elif self.calibrated:
                self.feedback = "Reset and hold for next jump phase"
            
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


class PushUpTest(FitnessTest):
    """Push-up test scored on full eccentric depth (elbow flexion) and concentric lockout."""
    DOWN_ELBOW_ANGLE = 100    
    UP_ELBOW_ANGLE = 160      
    FULL_DEPTH_ANGLE = 70.0   
    SHALLOW_DEPTH_ANGLE = 100.0
    LINE_TOLERANCE = 35.0     

    REQUIRED_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    def __init__(self):
        super().__init__()
        self.stage = "up"
        self.last_angles = {}
        self.peak_min_elbow = 180.0
        self.peak_line_angle = 180.0
        self.live_depth_pct = 0.0

    def _run(self, landmarks, frame_shape):
        if self._avg_visibility(landmarks, self.REQUIRED_LANDMARKS) < self.MIN_VISIBILITY:
            self.feedback = "Make sure your shoulders, elbows, hips and ankles are all visible"
            return self.quality_score, self.feedback

        L = mp_pose.PoseLandmark

        def pt(name, side):
            return get_xy(landmarks, getattr(L, f"{side}_{name}"))

        elbow_angles = {
            s: calculate_angle(pt("SHOULDER", s), pt("ELBOW", s), pt("WRIST", s)) 
            for s in ("LEFT", "RIGHT")
        }
        avg_elbow = (elbow_angles["LEFT"] + elbow_angles["RIGHT"]) / 2

        line_angles = [calculate_angle(pt("SHOULDER", s), pt("HIP", s), pt("ANKLE", s)) for s in ("LEFT", "RIGHT")]
        avg_line = sum(line_angles) / len(line_angles)

        self.last_angles = {
            "left_elbow": round(elbow_angles["LEFT"], 1),
            "right_elbow": round(elbow_angles["RIGHT"], 1),
            "body_line": round(avg_line, 1),
        }

        self.live_depth_pct = 100 * clamp(
            (self.SHALLOW_DEPTH_ANGLE - avg_elbow) / (self.SHALLOW_DEPTH_ANGLE - self.FULL_DEPTH_ANGLE), 0, 1
        )

        if avg_elbow < self.DOWN_ELBOW_ANGLE:
            self.stage = "down"
            self.peak_min_elbow = min(self.peak_min_elbow, avg_elbow)
            if abs(180 - avg_line) > abs(180 - self.peak_line_angle):
                self.peak_line_angle = avg_line
            self.feedback = "Full eccentric depth. Initiate concentric drive!" if abs(180 - avg_line) < self.LINE_TOLERANCE else "Warning: Lumbar hyperextension (hips sagging)"
        elif avg_elbow > self.UP_ELBOW_ANGLE and self.stage == "down":
            self.stage = "up"
            self._score_rep(self.peak_min_elbow, self.peak_line_angle)
            self.feedback = "Concentric lockout achieved."
            self.peak_min_elbow = 180.0
            self.peak_line_angle = 180.0
        elif self.stage == "up":
            self.feedback = "Control the eccentric descent."

        return self.quality_score, self.feedback

    def _score_rep(self, peak_min_elbow, peak_line_angle):
        depth_component = 60 * clamp(
            (self.SHALLOW_DEPTH_ANGLE - peak_min_elbow) / (self.SHALLOW_DEPTH_ANGLE - self.FULL_DEPTH_ANGLE), 0, 1
        )
        line_component = 40 * max(0.0, 1 - abs(180 - peak_line_angle) / self.LINE_TOLERANCE)
        self._record_rep(depth_component + line_component)

    def snapshot_extra(self):
        return {
            "angles": self.last_angles,
            "reps": self.reps,
            "depth_pct": round(self.live_depth_pct, 1),
        }


class PlankTest(FitnessTest):
    """Timed plank hold, tracking isometric stability and pelvic tilt."""
    LINE_TOLERANCE = 35.0
    SAMPLE_INTERVAL = 1.0      
    STABILITY_WINDOW = 15      
    STABILITY_TOLERANCE = 0.015

    REQUIRED_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    def __init__(self, duration=30.0):
        super().__init__()
        self.duration = duration
        self.start_time = None
        self.last_sample_time = None
        self.hip_y_history = []
        self.last_angles = {}
        self.live_form_pct = 0.0

    def start(self):
        super().start()
        self.start_time = time.time()
        self.last_sample_time = self.start_time

    def _run(self, landmarks, frame_shape):
        elapsed = time.time() - self.start_time
        remaining = max(0.0, self.duration - elapsed)

        if self._avg_visibility(landmarks, self.REQUIRED_LANDMARKS) < self.MIN_VISIBILITY:
            self.feedback = "Make sure your shoulders, hips and ankles are all visible"
            return self.quality_score, self.feedback

        L = mp_pose.PoseLandmark

        def pt(name, side):
            return get_xy(landmarks, getattr(L, f"{side}_{name}"))

        line_angles = [calculate_angle(pt("SHOULDER", s), pt("HIP", s), pt("ANKLE", s)) for s in ("LEFT", "RIGHT")]
        avg_line = sum(line_angles) / len(line_angles)
        self.last_angles = {"body_line": round(avg_line, 1)}

        avg_hip_y = (pt("HIP", "LEFT")[1] + pt("HIP", "RIGHT")[1]) / 2
        self.hip_y_history.append(avg_hip_y)
        
        if len(self.hip_y_history) > self.STABILITY_WINDOW:
            self.hip_y_history.pop(0)
            
        wobble = float(np.std(self.hip_y_history)) if len(self.hip_y_history) >= 3 else 0.0
        line_score = 100 * max(0.0, 1 - abs(180 - avg_line) / self.LINE_TOLERANCE)
        self.live_form_pct = line_score

        if abs(180 - avg_line) > self.LINE_TOLERANCE * 0.6:
            self.feedback = "Correct pelvic tilt: maintain rigid neutral spine"
        else:
            self.feedback = f"Optimal isometric core stability. {remaining:.1f}s"

        now = time.time()
        if now - self.last_sample_time >= self.SAMPLE_INTERVAL:
            self.last_sample_time = now
            stability_score = 100 * max(0.0, 1 - wobble / self.STABILITY_TOLERANCE)
            sample_score = 0.7 * line_score + 0.3 * stability_score
            self._record_rep(sample_score)

        if remaining <= 0 and not self.is_done:
            self.is_done = True
            self.feedback = f"Done! Hold form score: {self.quality_score}/100 over {self.duration:.0f}s"

        return self.quality_score, self.feedback

    def snapshot_extra(self):
        elapsed = 0.0 if self.start_time is None else time.time() - self.start_time
        return {
            "angles": self.last_angles,
            "reps": self.reps,  
            "form_pct": round(self.live_form_pct, 1),
            "time_remaining": round(max(0.0, self.duration - elapsed), 1),
        }


TEST_REGISTRY = {
    "running_spot": lambda: RunningInPlaceTest(duration=10.0),
    "jump": lambda: JumpTest(duration=15.0),
    "high_knees": lambda: HighKneeTest(duration=10.0),
    "pushup": lambda: PushUpTest(),
    "plank": lambda: PlankTest(duration=30.0),
}


# --- LIVE STATE MANAGER ---
class LiveState:
    def __init__(self):
        self._lock = threading.Lock()
        self.is_streaming = False
        self.stream_active = False
        self.camera_error = False
        self.last_client_ping = time.time()
        
        self.active_test = None
        self.test_name = "none"
        self.current_score = 0
        self.current_feedback = "Waiting to start..."
        self.current_extra = {}
        self.is_done = False

    def ping(self):
        """Acknowledges the frontend is still actively connected."""
        self.last_client_ping = time.time()

    def set_test(self, test_name):
        with self._lock:
            self.test_name = test_name
            self.current_score = 0
            self.current_feedback = "System ready. Awaiting manual start."
            self.current_extra = {}
            self.is_done = False
            factory = TEST_REGISTRY.get(test_name)
            self.active_test = factory() if factory else None

    def start_test(self):
        with self._lock:
            if self.active_test:
                self.active_test.start()
                self.current_feedback = self.active_test.feedback

    def restart_test(self):
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
                "camera_error": self.camera_error
            }
            payload.update(self.current_extra)
            return payload


live_state = LiveState()


def generate_live_stream():
    """Generator providing JPEG frames of the webcam with overlaid MediaPipe tracking."""
    live_state.is_streaming = True
    live_state.stream_active = True
    live_state.camera_error = False
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        live_state.camera_error = True
        live_state.is_streaming = False
        live_state.stream_active = False
        return

    try:
        while cap.isOpened() and live_state.is_streaming:
            # AUTO-KILL: If the frontend has vanished (e.g. page refresh), abort the camera loop
            if time.time() - live_state.last_client_ping > 4.0:
                print("Client disconnected. Releasing camera hardware.")
                break

            ret, frame = cap.read()
            if not ret:
                live_state.camera_error = True
                break

            frame = cv2.flip(frame, 1)  
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Keep execution thread-safe inside the lock to avoid test swapping mid-calculation
            with live_state._lock:
                test = live_state.active_test
                
                if results.pose_landmarks and test and not test.is_done:
                    score, feedback = test.process_frame(results.pose_landmarks.landmark, frame.shape)
                    
                    live_state.current_score = score
                    live_state.current_feedback = feedback
                    live_state.current_extra = test.snapshot_extra()
                    live_state.is_done = test.is_done

                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    finally:
        cap.release()
        pose.close()
        live_state.stream_active = False
        live_state.is_streaming = False


def process_video_file(input_path, output_path, mode):
    """Run an uploaded video through the given test end-to-end: writes an
    annotated (skeleton-overlaid) copy to output_path and returns the final
    score/feedback plus a per-frame timeline so the frontend can sync the
    rep counter and gauges to video playback instead of showing a single
    frozen final number.

    This runs against its own isolated FitnessTest instance rather than the
    shared `live_state`, so processing an upload never races with (or gets
    overwritten by) live webcam telemetry.

    Test duration cutoffs are measured against the video's own timestamps
    (frame_index / fps) rather than wall-clock time, since processing runs
    faster or slower than real playback depending on the machine. Skeleton
    drawing continues for every frame with a detected pose, even after the
    timed portion of the test ends, so the replay doesn't go blank partway
    through.
    """
    factory = TEST_REGISTRY.get(mode)
    if factory is None:
        raise ValueError(f"Unknown mode '{mode}'")
    test = factory()
    test.start()

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        pose.close()
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        pose.close()
        raise ValueError("Could not create annotated output video (codec unavailable)")

    timeline = []
    frame_index = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_index / fps
            if getattr(test, "start_time", None) is not None:
                test.start_time = time.time() - video_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                test.process_frame(results.pose_landmarks.landmark, frame.shape)
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            timeline.append({
                "t": round(video_time, 2),
                "test": mode,
                "score": test.quality_score,
                "feedback": test.feedback,
                "done": test.is_done,
                "started": True,
                **test.snapshot_extra(),
            })

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        pose.close()

    test.is_done = True
    final = {
        "test": mode,
        "score": test.quality_score,
        "feedback": test.feedback,
        "done": True,
        "started": True,
        **test.snapshot_extra(),
    }
    return {"final": final, "timeline": timeline}
