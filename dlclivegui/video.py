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
from tqdm import tqdm


def create_labeled_video(video_file,
                         ts_file,
                         dlc_file,
                         out_file=None,
                         save_images=False,
                         cut=(0, np.Inf),
                         cmap='bgy',
                         radius=3,
                         lik_thresh=0.5,
                         write_ts=False,
                         write_scale=2,
                         display=False,
                         progress=True,
                         label=True):
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
    save_images : bool, optional
        boolean flag to save still images in a folder
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
    progress : bool, optional
        boolean flag to display progress bar

    Raises
    ------
    Exception
        if frames cannot be read from the video file
    """

    cap = cv2.VideoCapture(video_file)
    cam_frame_times = np.load(ts_file)
    n_frames = cam_frame_times.size

    
    lab = "LABELED" if label else "UNLABELED"
    if out_file:
        out_file = f"{out_file}_VIDEO_{lab}.avi"
        out_times_file = f"{out_file}_TS_{lab}.npy"
    else:
        out_file = f"{os.path.splitext(video_file)[0]}_{lab}.avi"
        out_times_file = f"{os.path.splitext(ts_file)[0]}_L{lab}.npy"
    
    if save_images:
        im_dir = os.path.splitext(out_file)[0]
        os.makedirs(im_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = cap.get(cv2.CAP_PROP_FPS)
    im_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vwriter = cv2.VideoWriter(out_file, fourcc, fps, im_size)
    label_times = []

    if write_ts:
        ts_font = cv2.FONT_HERSHEY_PLAIN
        ts_coord = (0, im_size[1])
        ts_color = (255, 255, 255)
        ts_size = 2

    poses = pd.read_hdf(dlc_file)
    pose_times = poses['pose_time']
    poses = poses.melt(id_vars=['frame_time', 'pose_time'])
    bodyparts = poses['bodyparts'].unique()

    all_colors = getattr(cc, cmap)
    colors = [ImageColor.getcolor(c, "RGB")[::-1] for c in all_colors[::int(len(all_colors)/bodyparts.size)]]

    ind = 0
    vid_time = 0
    while vid_time < cut[0]:

        cur_time = cam_frame_times[ind]
        vid_time = cur_time - cam_frame_times[0]
        ret, frame = cap.read()
        ind += 1
        
        if not ret:
            raise Exception(f"Could not read frame = {ind+1} at time = {cur_time-cam_frame_times[0]}.")
        
    
    frame_times_sub = cam_frame_times[(cam_frame_times-cam_frame_times[0] > cut[0]) & (cam_frame_times-cam_frame_times[0] < cut[1])]
    iterator = tqdm(range(ind, ind+frame_times_sub.size)) if progress else range(ind, ind+frame_times_sub.size)
    
    for i in iterator:

        cur_time = cam_frame_times[i]
        vid_time = cur_time - cam_frame_times[0]
        ret, frame = cap.read()
        
        if not ret:
            raise Exception(f"Could not read frame = {i+1} at time = {cur_time-cam_frame_times[0]}.")

        cur_pose_time = pose_times[np.where(pose_times - cur_time > 0)[0][0]]
        this_pose = poses[poses['pose_time']==cur_pose_time]

        if label:
            for j in range(bodyparts.size):
                this_bp = this_pose[this_pose['bodyparts'] == bodyparts[j]]['value'].values
                if this_bp[2] > lik_thresh:
                    x = int(this_bp[0])
                    y = int(this_bp[1])
                    frame = cv2.circle(frame, (x, y), radius, colors[j], thickness=-1)

        if write_ts:
            frame = cv2.putText(frame, f"{vid_time:0.3f}", ts_coord, ts_font, write_scale, ts_color, ts_size)

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
