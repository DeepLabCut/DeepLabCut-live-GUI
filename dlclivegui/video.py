"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import os
import numpy as np
import pandas as pd
import cv2
import colorcet as cc
from PIL import ImageColor


def create_labeled_video(video_file, ts_file, dlc_file, out_file=None, save_images=False, cut=(0, np.Inf), cmap='bgy', radius=3, lik_thresh=0.5, display=False):
    """ Create a labeled video from DeepLabCut-live-GUI recording

    Parameters
    ----------
    video_file : str 
        path to video file
    ts_file : str
        path to timestamps file
    dlc_file : str
        path to DeepLabCut file
    out_file : str, optional
        path for output file. If None, output file will be "'video_file'_LABELED.avi". by default None. If NOn
    cut : tuple, optional
        time of video to use. Will only save labeled video for time after cut[0] and before cut[1], by default (0, np.Inf)
    cmap : str, optional
        a :package:`colorcet` colormap, by default 'bgy'
    radius : int, optional
        radius for keypoints, by default 3
    lik_thresh : float, optional
        likelihood threshold to plot keypoints, by default 0.5
    display : bool, optional
        boolean flag to display images as video is written, by default False

    Raises
    ------
    Exception
        if frames cannot be read from the video file
    """

    cap = cv2.VideoCapture(video_file)
    cam_frame_times = np.load(ts_file)
    n_frames = cam_frame_times.size

    if out_file:
        out_file = f"{out_file}_VIDEO_LABELED.avi"
        out_times_file = f"{out_file}_TS_LABELED.npy"
    else:
        out_file = f"{os.path.splitext(video_file)[0]}_LABELED.avi"
        out_times_file = f"{os.path.splitext(ts_file)[0]}_LABELED.npy"
    
    if save_images:
        im_dir = os.path.splitext(out_file)[0]
        os.makedirs(im_dir)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = cap.get(cv2.CAP_PROP_FPS)
    im_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vwriter = cv2.VideoWriter(out_file, fourcc, fps, im_size)
    label_times = []

    poses = pd.read_hdf(dlc_file)
    pose_times = poses['pose_time']
    poses = poses.melt(id_vars=['frame_time', 'pose_time'])
    bodyparts = poses['bodyparts'].unique()

    all_colors = getattr(cc, cmap)
    colors = [ImageColor.getcolor(c, "RGB")[::-1] for c in all_colors[::int(len(all_colors)/bodyparts.size)]]

    for i in range(cam_frame_times.size):

        cur_time = cam_frame_times[i]
        vid_time = cur_time - cam_frame_times[0]
        ret, frame = cap.read()
        
        if not ret:
            raise Exception(f"Could not read frame = {i+1} at time = {cur_time-cam_frame_times[0]}.")

        if vid_time > cut[1]:

            break

        elif vid_time > cut[0]:

            cur_pose_time = pose_times[np.where(pose_times - cur_time > 0)[0][0]]
            this_pose = poses[poses['pose_time']==cur_pose_time]

            for j in range(bodyparts.size):
                this_bp = this_pose[this_pose['bodyparts'] == bodyparts[j]]['value'].values
                if this_bp[2] > lik_thresh:
                    x = int(this_bp[0])
                    y = int(this_bp[1])
                    frame = cv2.circle(frame, (x, y), radius, colors[j], thickness=-1)

            if display:
                cv2.imshow('DLC Live Labeled Video', frame)
                cv2.waitKey(1)
            
            vwriter.write(frame)
            label_times.append(cur_time)
            if save_images:
                new_file = f"{im_dir}/frame_{i}.png"
                cv2.imwrite(new_file, frame)

    if display:
        cv2.destroyAllWindows()
    
    vwriter.release()
    np.save(out_times_file, label_times)
