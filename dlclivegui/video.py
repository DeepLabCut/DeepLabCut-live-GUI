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


def create_labeled_video(
    data_dir,
    out_dir=None,
    dlc_online=True,
    save_images=False,
    cut=(0, np.Inf),
    crop=None,
    cmap="bmy",
    radius=3,
    lik_thresh=0.5,
    write_ts=False,
    write_scale=2,
    write_pos="bottom-left",
    write_ts_offset=0,
    display=False,
    progress=True,
    label=True,
):
    """ Create a labeled video from DeepLabCut-live-GUI recording

    Parameters
    ----------
    data_dir : str
        path to data directory
    dlc_online : bool, optional
        flag indicating dlc keypoints from online tracking, using DeepLabCut-live-GUI, or offline tracking, using :func:`dlclive.benchmark_videos`
    out_file : str, optional
        path for output file. If None, output file will be "'video_file'_LABELED.avi". by default None. If NOn
    save_images : bool, optional
        boolean flag to save still images in a folder
    cut : tuple, optional
        time of video to use. Will only save labeled video for time after cut[0] and before cut[1], by default (0, np.Inf)
    cmap : str, optional
        a :package:`colorcet` colormap, by default 'bmy'
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

    base_dir = os.path.basename(data_dir)
    video_file = os.path.normpath(f"{data_dir}/{base_dir}_VIDEO.avi")
    ts_file = os.path.normpath(f"{data_dir}/{base_dir}_TS.npy")
    dlc_file = (
        os.path.normpath(f"{data_dir}/{base_dir}_DLC.hdf5")
        if dlc_online
        else os.path.normpath(f"{data_dir}/{base_dir}_VIDEO_DLCLIVE_POSES.h5")
    )

    cap = cv2.VideoCapture(video_file)
    cam_frame_times = np.load(ts_file)
    n_frames = cam_frame_times.size

    lab = "LABELED" if label else "UNLABELED"
    if out_dir:
        out_file = (
            f"{out_dir}/{os.path.splitext(os.path.basename(video_file))[0]}_{lab}.avi"
        )
        out_times_file = (
            f"{out_dir}/{os.path.splitext(os.path.basename(ts_file))[0]}_{lab}.npy"
        )
    else:
        out_file = f"{os.path.splitext(video_file)[0]}_{lab}.avi"
        out_times_file = f"{os.path.splitext(ts_file)[0]}_{lab}.npy"

    os.makedirs(os.path.normpath(os.path.dirname(out_file)), exist_ok=True)

    if save_images:
        im_dir = os.path.splitext(out_file)[0]
        os.makedirs(im_dir, exist_ok=True)

    im_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if crop is not None:
        crop[0] = crop[0] if crop[0] > 0 else 0
        crop[1] = crop[1] if crop[1] > 0 else im_size[1]
        crop[2] = crop[2] if crop[2] > 0 else 0
        crop[3] = crop[3] if crop[3] > 0 else im_size[0]
        im_size = (crop[3] - crop[2], crop[1] - crop[0])

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    fps = cap.get(cv2.CAP_PROP_FPS)
    vwriter = cv2.VideoWriter(out_file, fourcc, fps, im_size)
    label_times = []

    if write_ts:
        ts_font = cv2.FONT_HERSHEY_PLAIN

        if "left" in write_pos:
            ts_w = 0
        else:
            ts_w = (
                im_size[0] if crop is None else (crop[3] - crop[2]) - (55 * write_scale)
            )

        if "bottom" in write_pos:
            ts_h = im_size[1] if crop is None else (crop[1] - crop[0])
        else:
            ts_h = 0 if crop is None else crop[0] + (12 * write_scale)

        ts_coord = (ts_w, ts_h)
        ts_color = (255, 255, 255)
        ts_size = 2

    poses = pd.read_hdf(dlc_file)
    if dlc_online:
        pose_times = poses["pose_time"]
    else:
        poses["frame_time"] = cam_frame_times
        poses["pose_time"] = cam_frame_times
    poses = poses.melt(id_vars=["frame_time", "pose_time"])
    bodyparts = poses["bodyparts"].unique()

    all_colors = getattr(cc, cmap)
    colors = [
        ImageColor.getcolor(c, "RGB")[::-1]
        for c in all_colors[:: int(len(all_colors) / bodyparts.size)]
    ]

    ind = 0
    vid_time = 0
    while vid_time < cut[0]:

        cur_time = cam_frame_times[ind]
        vid_time = cur_time - cam_frame_times[0]
        ret, frame = cap.read()
        ind += 1

        if not ret:
            raise Exception(
                f"Could not read frame = {ind+1} at time = {cur_time-cam_frame_times[0]}."
            )

    frame_times_sub = cam_frame_times[
        (cam_frame_times - cam_frame_times[0] > cut[0])
        & (cam_frame_times - cam_frame_times[0] < cut[1])
    ]
    iterator = (
        tqdm(range(ind, ind + frame_times_sub.size))
        if progress
        else range(ind, ind + frame_times_sub.size)
    )
    this_pose = np.zeros((bodyparts.size, 3))

    for i in iterator:

        cur_time = cam_frame_times[i]
        vid_time = cur_time - cam_frame_times[0]
        ret, frame = cap.read()

        if not ret:
            raise Exception(
                f"Could not read frame = {i+1} at time = {cur_time-cam_frame_times[0]}."
            )

        if dlc_online:
            poses_before_index = np.where(pose_times < cur_time)[0]
            if poses_before_index.size > 0:
                cur_pose_time = pose_times[poses_before_index[-1]]
                this_pose = poses[poses["pose_time"] == cur_pose_time]
        else:
            this_pose = poses[poses["frame_time"] == cur_time]

        if label:
            for j in range(bodyparts.size):
                this_bp = this_pose[this_pose["bodyparts"] == bodyparts[j]][
                    "value"
                ].values
                if this_bp[2] > lik_thresh:
                    x = int(this_bp[0])
                    y = int(this_bp[1])
                    frame = cv2.circle(frame, (x, y), radius, colors[j], thickness=-1)

        if crop is not None:
            frame = frame[crop[0] : crop[1], crop[2] : crop[3]]

        if write_ts:
            frame = cv2.putText(
                frame,
                f"{(vid_time-write_ts_offset):0.3f}",
                ts_coord,
                ts_font,
                write_scale,
                ts_color,
                ts_size,
            )

        if display:
            cv2.imshow("DLC Live Labeled Video", frame)
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


def main():

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str, default=None)
    parser.add_argument("--dlc-offline", action="store_true")
    parser.add_argument("-s", "--save-images", action="store_true")
    parser.add_argument("-u", "--cut", nargs="+", type=float, default=[0, np.Inf])
    parser.add_argument("-c", "--crop", nargs="+", type=int, default=None)
    parser.add_argument("-m", "--cmap", type=str, default="bmy")
    parser.add_argument("-r", "--radius", type=int, default=3)
    parser.add_argument("-l", "--lik-thresh", type=float, default=0.5)
    parser.add_argument("-w", "--write-ts", action="store_true")
    parser.add_argument("--write-scale", type=int, default=2)
    parser.add_argument("--write-pos", type=str, default="bottom-left")
    parser.add_argument("--write-ts-offset", type=float, default=0.0)
    parser.add_argument("-d", "--display", action="store_true")
    parser.add_argument("--no-progress", action="store_false")
    parser.add_argument("--no-label", action="store_false")
    args = parser.parse_args()

    create_labeled_video(
        args.dir,
        out_dir=args.out_dir,
        dlc_online=(not args.dlc_offline),
        save_images=args.save_images,
        cut=tuple(args.cut),
        crop=args.crop,
        cmap=args.cmap,
        radius=args.radius,
        lik_thresh=args.lik_thresh,
        write_ts=args.write_ts,
        write_scale=args.write_scale,
        write_pos=args.write_pos,
        write_ts_offset=args.write_ts_offset,
        display=args.display,
        progress=args.no_progress,
        label=args.no_label,
    )
