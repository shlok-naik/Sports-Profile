from flask import Flask, Response, jsonify, render_template, request
from detector import generate_live_stream, live_state

app = Flask(__name__)

# Temporary memory for the user's combine stats
user_stats = {"squat": 0, "high_knees": 0}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_live_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    mode = data.get("mode", "squat")
    live_state.set_test(mode)
    return jsonify({"status": "success", "mode": mode})

@app.route('/live_stats')
def live_stats():
    return jsonify(live_state.snapshot())

@app.route('/save_score', methods=['POST'])
def save_score():
    # Save the score when a test finishes
    snap = live_state.snapshot()
    if snap["test"] in user_stats:
        user_stats[snap["test"]] = snap["score"]
    return jsonify({"status": "saved", "stats": user_stats})

@app.route('/generate_profile', methods=['GET'])
def generate_profile():
    squat = user_stats["squat"]
    run = user_stats["high_knees"]
    
    # Scout Profile Logic
    sport = "Undecided"
    skills = []
    
    if squat > 20 and run > 30:
        sport = "American Football / Rugby"
        skills = ["High Motor", "Elite Leg Drive", "Conditioned"]
    elif run > 35:
        sport = "Track & Field / Soccer"
        skills = ["Speed Demon", "High Cadence", "Agility"]
    elif squat > 25:
        sport = "Powerlifting / Linebacker"
        skills = ["Raw Power", "Explosive Base"]
    else:
        sport = "Developing Prospect"
        skills = ["Needs time in the lab", "Solid Foundation"]

    return jsonify({
        "sport": sport,
        "skills": skills,
        "squat_stat": squat,
        "run_stat": run
    })

if __name__ == '__main__':
    # Initialize default
    live_state.set_test("high_knees")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)