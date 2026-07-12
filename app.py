from flask import Flask, Response, jsonify, render_template, request
from detector import generate_live_stream, live_state, TEST_REGISTRY

app = Flask(__name__)

# Every test now saves its composite 0-100 form score alongside supporting
# numbers (rep/step/sample count, or best jump distance).
user_stats = {
    "running_spot": {"score": 0, "reps": 0},
    "high_knees": {"score": 0, "reps": 0},
    "jump": {"score": 0, "best_cm": 0},
    "pushup": {"score": 0, "reps": 0},
    "plank": {"score": 0, "reps": 0},
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_live_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json or {}
    mode = data.get("mode", "high_knees")
    if mode not in TEST_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown mode '{mode}'"}), 400
    live_state.set_test(mode)
    return jsonify({"status": "initialized", "mode": mode})


@app.route('/start_test', methods=['POST'])
def start_test():
    live_state.start_test()
    return jsonify({"status": "started", "snapshot": live_state.snapshot()})


@app.route('/restart_test', methods=['POST'])
def restart_test():
    live_state.restart_test()
    return jsonify({"status": "restarted", "snapshot": live_state.snapshot()})


@app.route('/live_stats')
def live_stats():
    return jsonify(live_state.snapshot())


@app.route('/save_score', methods=['POST'])
def save_score():
    snap = live_state.snapshot()
    test = snap.get("test")

    if test == "jump":
        user_stats["jump"] = {
            "score": snap.get("score", 0),
            "best_cm": snap.get("best_jump_cm", 0),
        }
    elif test in user_stats:
        user_stats[test] = {
            "score": snap.get("score", 0),
            "reps": snap.get("reps", 0),
        }

    return jsonify({"status": "saved", "saved_data": user_stats})


@app.route('/generate_profile', methods=['GET'])
def generate_profile():
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


if __name__ == '__main__':
    live_state.set_test("running_spot")

    print("\n🚀 Combine AI Dashboard is live! Open your browser and go to:")
    print("👉 http://127.0.0.1:5001\n")

    app.run(host='127.0.0.1', port=5001, debug=False)