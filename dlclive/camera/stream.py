import matplotlib
matplotlib.use('TkAgg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import LifoQueue
from time import time


class StreamReader(threading.Thread):
    def __init__(self, video_path, max_queue_size=20):
        super(StreamReader, self).__init__(target=self.read, daemon=True)
        self.video = cv2.VideoCapture(video_path)
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.queue = LifoQueue(maxsize=max_queue_size)
        self.streaming = True

    def read(self):
        while self.streaming:
            if not self.queue.full():
                _, frame = self.video.read()
                if frame is not None:
                    self.queue.put(frame[:, :, ::-1])
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


class StreamDisplay(FigureCanvasTkAgg):
    def __init__(self, stream, parent, dpi=100, **kwargs):
        self.stream = stream
        self.parent = parent

        figsize = self.stream.width // dpi, self.stream.height // dpi
        self.fig = plt.Figure(frameon=False, figsize=figsize, dpi=dpi)
        super(StreamDisplay, self).__init__(self.fig, master=parent)
        self.ax = self.fig.add_subplot(111)
        img = np.random.random((self.stream.height, self.stream.width, 3))
        self.im = self.ax.imshow(img, interpolation='none')
        nbodyparts = 4
        cmap = plt.cm.get_cmap('winter', nbodyparts)
        colors = cmap(range(nbodyparts))
        self.scat = self.ax.scatter([], [], **kwargs)
        self.scat.set_color(colors)
        self.ax.set_xlim(0, self.stream.width)
        self.ax.set_ylim(0, self.stream.height)
        self.ax.axis('off')
        self.ax.invert_yaxis()
        self.draw()
        self.background = self.copy_from_bbox(self.ax.bbox)

    def refresh(self, ms=10):
        if not self.stream.queue.empty():
            frame = self.stream.queue.get()
            self.im.set_data(frame)
            points = np.c_[np.random.randint(0, self.stream.height, 4),
                           np.random.randint(0, self.stream.width, 4)]
            self.scat.set_offsets(points)
            self.restore_region(self.background)
            self.ax.draw_artist(self.im)
            self.ax.draw_artist(self.scat)
            self.blit(self.ax.bbox)
        self.parent.after(ms, self.refresh)


class StreamWindow(tk.Toplevel):
    def __init__(self, stream, parent, dpi=100, **kwargs):
        super(StreamWindow, self).__init__(parent)
        self.stream = stream
        self.parent = parent
        self.lift()
        self.attributes('-topmost', True)
        self.display = StreamDisplay(self.stream, self, dpi, **kwargs)
        self.display.refresh()
        self.start_button = tk.Button(self, text='Start', command=self.stream.start)
        self.display.get_tk_widget().grid(row=0, column=0)
        self.start_button.grid(row=1, column=0)


if __name__ == '__main__':
    video = '/Users/Jessy/Documents/PycharmProjects/lactic/videos_hq/davidrudisha.mp4'
    stream = StreamReader(video)
    root = tk.Tk()
    window = StreamWindow(stream, root, s=12, alpha=0.7)
    root.mainloop()