import cv2
import numpy as np
import pyqtgraph as pg
import sys
import threading
from collections import deque
from pyqtgraph.Qt import QtCore, QtWidgets
from time import time


class StreamReader(threading.Thread):
    def __init__(self, video_path, max_queue_size=100):
        super(StreamReader, self).__init__(target=self.read, daemon=True)
        self.video = cv2.VideoCapture(video_path)
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.queue = deque(maxlen=max_queue_size)
        self.streaming = True

    def read(self):
        while self.streaming:
            _, frame = self.video.read()
            if frame is not None:
                self.queue.append(frame[:, :, ::-1])
            else:
                self.stop()

    def stop(self):
        self.streaming = False

    def test_speed(self):
        start = time()
        success = True
        while success:
            success, frame = self.video.read()
        end = time()
        print(f'fps={self.nframes / (end - start)}')


class FastStreamDisplay(pg.GraphicsLayoutWidget):
    def __init__(self, stream, parent=None):
        super(FastStreamDisplay, self).__init__(parent)
        # self.ci.layout.setContentsMargins(0, 0, 0, 0)
        # self.ci.layout.setSpacing(0)
        self.stream = stream
        self.parent = parent
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0, 0, self.stream.width, self.stream.height))

        self.img = pg.ImageItem()
        self.points = pg.ScatterPlotItem()
        self.view.addItem(self.img)
        self.view.addItem(self.points)

        self.fps = 0.
        self.last_update = time()

    def refresh(self):
        if self.stream.queue:
            frame = self.stream.queue.pop()
            self.img.setImage(np.rot90(frame, axes=(1, 0)))
            points = np.c_[np.random.randint(0, self.stream.height, 4),
                           np.random.randint(0, self.stream.width, 4)]
            self.points.setData(pos=points)

        now = time()
        dt = now - self.last_update
        fps2 = 1.0 / max(dt, 0.000000000001)
        self.last_update = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
        self.parent.fps_label.setText(tx)
        QtCore.QTimer.singleShot(1, self.refresh)


class FastStreamWindow(QtWidgets.QMainWindow):
    def __init__(self, stream, parent=None):
        super(FastStreamWindow, self).__init__(parent)
        self.setWindowTitle('DeepLabCut Live')
        self.resize(stream.width, stream.height)

        self.mainbox = QtWidgets.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtWidgets.QVBoxLayout())

        self.fps_label = QtWidgets.QLabel()
        self.canvas = FastStreamDisplay(stream, self)
        self.mainbox.layout().addWidget(self.canvas)
        self.mainbox.layout().addWidget(self.fps_label)

        self.canvas.refresh()
        self.canvas.stream.start()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # video = '/home/jessy/PycharmProjects/DLCdev/test-jessy-2020-01-28/videos/reachingvideo1.avi'
    video = '/home/jessy/PycharmProjects/DLCdev/datasets/Horses-Byron-2019-05-08/videos/output4.mp4'
    # video = '/home/jessy/PycharmProjects/DLCdev/MontBlanc-Daniel-2020-01-29/videos/montblanc.mov'
    stream = StreamReader(video)
    thisapp = FastStreamWindow(stream)
    thisapp.show()
    sys.exit(app.exec_())
