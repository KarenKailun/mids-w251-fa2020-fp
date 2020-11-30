import cv2
import threading
import time

from collections import deque
from flask import Flask, render_template, Response
from ..detector import BreathRateDetector

#Initialize the Flask app
app = Flask(__name__)

class InProgress(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._in_progress = False
        self._point_buffer = deque([], maxlen=440)

    def start_progress(self):
        with self._lock:
            if not self._in_progress:
                self._in_progress = True
                self._point_buffer.clear()
                return True
            # return false to indicate progress had already been started
            return False
    
    def stop_progress(self):
        with self._lock:
            if self._in_progress:
                self._in_progress = False
                return True
            return False

    def in_progress(self):
        with self._lock:
            return self._in_progress

    def append(self, to_add):
        with self._lock:
            if self._in_progress:
                self._point_buffer.append(to_add)
                return True
            return False

    def get_buffer(self):
        with self._lock:
            return self._point_buffer.copy()

class LockedCamera(object):
    def __init__(self, camera_id):
        self._lock = threading.Lock()
        print('Creating capture with camera ID: {}'.format(camera_id))
        self._camera = cv2.VideoCapture(camera_id)
        self._camera.set(cv2.CAP_PROP_FPS, 10) # camera should capture at 10fps 

    def get_frame(self):
        with self._lock:
            success, frame = self._camera.read()
            return (success, frame)

    def is_open(self):
        with self._lock:
            return self._camera.isOpened()

    def _close(self, manual_close):
        self._camera.release()
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._close(manual_close=False)

capture_status = InProgress()
detector = BreathRateDetector()
frame_queue = deque([], maxlen=1000)
loop = None

def camera_loop():
    camera = LockedCamera(1)
    print('Starting camera loop...')
    while camera.is_open():
        success, frame = camera.get_frame()  # read the camera frame
        if not success:
            break
        else:
            if capture_status.in_progress():
                points = detector.execute(frame)
                if len(points) > 0:
                    frame = detector.draw_keypoints(frame, points)
                capture_status.append(points)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            frame_queue.appendleft(frame)
    print('Camera loop exited...')

def gen_frames():
    print('Starting frame gen...')  
    while True:
        try:
            frame = frame_queue.pop()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except IndexError:
            time.sleep(.025)



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/start_capture')
def start_capture():
    return {'started': capture_status.start_progress()}

@app.route('/end_capture')
def end_capture():
    # todo: send to mqtt
    return {'ended': capture_status.stop_progress()}

@app.route('/get_last_rate')
def get_last_rate():
    points = capture_status.get_buffer()
    rate = {'rate': detector.rate_from_keypoints(points)}
    return rate

if __name__ == "__main__":
    print('OpenCV2 version: {}'.format(cv2.__version__))
    print('Running app...')

    loop = threading.Thread(target=camera_loop, daemon=True)
    loop.start()

    app.run(debug=False, host='192.168.1.197')
