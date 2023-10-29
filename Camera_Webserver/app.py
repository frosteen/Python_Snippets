import cv2
from flask import Flask, Response, render_template
from imutils.video import VideoStream

app = Flask(__name__)

capture = None


def gen_frames():
    global capture
    while True:
        frame = capture.read()

        # continue loop if frame does not exist
        if frame is None:
            continue

        # convert frame to buffer
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        # concat frame one by one and show result
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    global capture
    if capture is None:
        # Using VideoStream from imutils (fast because of threading)
        capture = VideoStream(src=0).start()

    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
