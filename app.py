"""
Flask application handling API routing and bridging the frontend to OpenCV tracking.
"""
import os
import time
import uuid
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename
from detector import generate_live_stream, live_state, TEST_REGISTRY, process_video_file
from database import save_athlete, get_athletes, create_tables

app = Flask(__name__)

UPLOAD_DIR = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm', 'mkv', 'm4v'}


def _is_allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# User stats (Global). 
user_stats = {
    "running_spot": {"score": 0, "reps": 0},
    "high_knees": {"score": 0, "reps": 0},
    "jump": {"score": 0, "best_cm": 0},
    "pushup": {"score": 0, "reps": 0},
    "plank": {"score": 0, "reps": 0},
}


def calculate_sport():

    jump = user_stats["jump"]["score"]
    run = user_stats["running_spot"]["score"]
    push = user_stats["pushup"]["score"]
    plank = user_stats["plank"]["score"]


    if jump >= 80:
        return "Basketball / Volleyball"

    elif run >= 80:
        return "Sprinting / Soccer"

    elif push >= 80:
        return "Strength Sports"

    elif plank >= 80:
        return "Gymnastics / CrossFit"

    return "General Athlete"


@app.route('/')
def index():
    """Serves the main frontend dashboard."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Streams the OpenCV video feed using multipart replacement."""
    
    # If a feed already exists (e.g. from a page refresh), force it to terminate
    # before spinning up a new one so they don't fight over webcam access.
    if live_state.stream_active:
        live_state.is_streaming = False
        start_wait = time.time()
        # Wait up to 2 seconds for the old camera thread to release hardware
        while live_state.stream_active and time.time() - start_wait < 2.0:
            time.sleep(0.1)
            
    return Response(
        generate_live_stream(), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Changes the active fitness test mode."""
    data = request.json or {}
    mode = data.get("mode", "high_knees")
    
    if mode not in TEST_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown mode '{mode}'"}), 400
        
    live_state.set_test(mode)
    return jsonify({"status": "initialized", "mode": mode})


@app.route('/start_test', methods=['POST'])
def start_test():
    """Triggers the active test to begin tracking."""
    live_state.start_test()
    return jsonify({"status": "started", "snapshot": live_state.snapshot()})


@app.route('/restart_test', methods=['POST'])
def restart_test():
    """Resets the currently active test data."""
    live_state.restart_test()
    return jsonify({"status": "restarted", "snapshot": live_state.snapshot()})


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Processes an uploaded video file end-to-end (instead of the live webcam)
    and returns the final score plus a URL to the annotated replay."""
    file = request.files.get('video')
    mode = request.form.get('mode', live_state.test_name)

    if not file or file.filename == '':
        return jsonify({"status": "error", "message": "No video file provided"}), 400
    if not _is_allowed_video(file.filename):
        return jsonify({"status": "error", "message": "Unsupported video format"}), 400
    if mode not in TEST_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown mode '{mode}'"}), 400

    uid = uuid.uuid4().hex
    upload_path = os.path.join(UPLOAD_DIR, f"{uid}_{secure_filename(file.filename)}")
    file.save(upload_path)

    output_filename = f"annotated_{uid}.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    try:
        process_video_file(upload_path, output_path, mode)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not process video: {e}"}), 500
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

    snap = live_state.snapshot()
    snap["video_url"] = f"/static/uploads/{output_filename}"
    return jsonify({"status": "processed", **snap})


@app.route('/live_stats')
def live_stats():
    """
    Returns real-time telemetry from the OpenCV processing thread.
    Also acts as a heartbeat check to keep the webcam stream alive.
    """
    live_state.ping()
    return jsonify(live_state.snapshot())


@app.route('/save_score', methods=['POST'])
def save_score():

    snap = live_state.snapshot()

    test = snap.get("test")


    if test == "jump":

        user_stats["jump"] = {
            "score": snap.get("score",0),
            "best_cm": snap.get("best_jump_cm",0)
        }


    elif test in user_stats:

        user_stats[test] = {
            "score": snap.get("score",0),
            "reps": snap.get("reps",0)
        }



    existing = get_athletes()


    athlete = {

        "id": len(existing) + 1,

        "name": f"Athlete {len(existing)+1}",

        "running":
        user_stats["running_spot"]["score"],

        "high_knees":
        user_stats["high_knees"]["score"],

        "jump":
        user_stats["jump"]["best_cm"],

        "pushups":
        user_stats["pushup"]["score"],

        "plank":
        user_stats["plank"]["score"],

        "sport":
        calculate_sport()

    }


    save_athlete(athlete)


    return jsonify({

        "status":"saved",

        "athlete":athlete,

        "total":len(existing)+1

    })


@app.route('/generate_profile', methods=['GET'])
def generate_profile():
    """Generates an athletic profile based on saved metrics."""
    run_score = user_stats["running_spot"]["score"]
    knee_score = user_stats["high_knees"]["score"]
    jump_score = user_stats["jump"]["score"]
    jump_cm = user_stats["jump"]["best_cm"]
    pushup_score = user_stats["pushup"]["score"]
    plank_score = user_stats["plank"]["score"]

    scores = [run_score, knee_score, jump_score, pushup_score, plank_score]
    any_data = any(s > 0 for s in scores)

    recommended_sport = "General Athlete"
    traits = ["Adaptive Athlete"]

    # Trait logic tree
    if pushup_score >= 75 and plank_score >= 75 and knee_score >= 70:
        recommended_sport = "Multi-Sport / Combine All-Rounder"
        traits = ["Complete Athletic Profile", "Strong Core-to-Power Transfer"]
    elif jump_score >= 80:
        recommended_sport = "Basketball / Volleyball"
        traits = ["Elite Vertical Explosiveness", "Fast-Twitch Dominant"]
    elif run_score >= 80 and knee_score >= 70:
        recommended_sport = "Sprinting / Soccer"
        traits = ["Efficient Running Form", "High Cadence", "Agile Core"]
    elif pushup_score >= 80:
        recommended_sport = "Football (Line) / Combat Sports"
        traits = ["Elite Upper-Body Power", "Strong Pressing Strength"]
    elif plank_score >= 80:
        recommended_sport = "Gymnastics / CrossFit"
        traits = ["Elite Core Stability", "Excellent Body Control"]
    elif any_data:
        recommended_sport = "Developing Prospect"
        traits = ["Active Foundation", "Building Motor Skills"]

    return jsonify({
        "sport": recommended_sport,
        "skills": traits,
        "run_stat": run_score,
        "knee_stat": knee_score,
        "jump_stat": jump_cm,
        "jump_score": jump_score,
        "pushup_stat": pushup_score,
        "plank_stat": plank_score,
    })


@app.route('/scout')
def scout():

    return render_template(
        'scout.html'
    )


@app.route('/athletes')
def athletes_api():

    return jsonify(get_athletes())


if __name__ == '__main__':

    create_tables()

    live_state.set_test("running_spot")

    HOST = '127.0.0.1'
    PORT = 5001

    print(f"\n🚀 Combine AI Dashboard is live! Open your browser and go to:")
    print(f"👉 http://{HOST}:{PORT}\n")

    app.run(host=HOST, port=PORT, debug=False)