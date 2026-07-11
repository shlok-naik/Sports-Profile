from flask import Flask, Response, jsonify, render_template, request
from detector import generate_live_stream, live_state, TEST_REGISTRY

app = Flask(__name__)

# Each test now saves its composite 0-100 form score alongside supporting
# numbers (rep count, or best jump distance) rather than a raw rep tally.
user_stats = {
    "high_knees": {"score": 0, "reps": 0},
    "squat": {"score": 0, "reps": 0},
    "jump": {"score": 0, "best_cm": 0},
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
    squat_score = user_stats["squat"]["score"]
    run_score = user_stats["high_knees"]["score"]
    jump_score = user_stats["jump"]["score"]
    jump_cm = user_stats["jump"]["best_cm"]

    recommended_sport = "General Athlete"
    traits = ["Adaptive Athlete"]
    any_data = squat_score > 0 or run_score > 0 or jump_score > 0

    if squat_score >= 70 and run_score >= 70 and jump_cm >= 45:
        recommended_sport = "Rugby / American Football"
        traits = ["Powerhouse Base", "High Engine", "Explosive Accelerator"]
    elif jump_score >= 80:
        recommended_sport = "Basketball / Volleyball"
        traits = ["Elite Vertical Explosiveness", "Fast-Twitch Dominant"]
    elif run_score >= 80:
        recommended_sport = "Sprinting / Soccer"
        traits = ["Rapid Foot-Speed", "High Cadence Rate", "Agile Core"]
    elif squat_score >= 80:
        recommended_sport = "Combat Sports / Weightlifting"
        traits = ["Elite Lower-Body Power", "Solid Center of Gravity"]
    elif any_data:
        recommended_sport = "Developing Prospect"
        traits = ["Active Foundation", "Building Motor Skills"]

    return jsonify({
        "sport": recommended_sport,
        "skills": traits,
        "squat_stat": squat_score,
        "run_stat": run_score,
        "jump_stat": jump_cm,
        "jump_score": jump_score,
    })


if __name__ == '__main__':
    live_state.set_test("high_knees")

    print("\n🚀 Combine AI Dashboard is live! Open your browser and go to:")
    print("👉 http://127.0.0.1:5000\n")

    app.run(host='127.0.0.1', port=5000, debug=False)