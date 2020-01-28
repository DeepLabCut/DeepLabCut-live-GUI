"""
GUI to run DLC live
Copyright M. Mathis Lab
Written by  Gary Kane - https://github.com/gkane26
post-doctoral fellow @ the Adaptive Motor Control Lab
https://github.com/AdaptiveMotorControlLab
"""

from tkinter import Tk, Label, Entry, Button, Radiobutton, StringVar, IntVar, filedialog, messagebox
from tkinter.ttk import Combobox
import tkfilebrowser
import os
import json
import numpy as np
import time
import datetime
import cv2
import pandas as pd
import pickle
import importlib
import inspect

import camera
import processor
from deeplabcut.pose_estimation_tensorflow import DLCLive

# from tkinter import Entry, Label, Button, StringVar, IntVar, Tk, END, Radiobutton, filedialog, ttk
# import numpy as np
# from ic_camera import ICCam
# import time
# import ffmpy
# import threading
# import nidaqmx

class DLCLiveGUI(object):

    def __init__(self):

        ### get configuration ###

        self.cfg = self.get_config()

        ### get all camera names ###

        # self.all_camera_names = ()
        # for i in self.cfg['camera_list']:
        #     self.all_camera_names = self.all_camera_names + (i['name'],)
        self.get_camera_names()
        self.cam = None
        self.vid_out = None
        self.dlc_live = None
        self.proc = None

        ### create GUI window ###

        self.createGUI()

    def get_camera_names(self):
        self.all_camera_names = ()
        for i in self.cfg['camera_list']:
            self.all_camera_names = self.all_camera_names + (i['name'],)

    def get_config(self):

        ### read configuration file ###

        path = os.path.dirname(os.path.realpath(__file__))
        self.cfg_file = os.path.normpath(path + '/config.json')
        if os.path.isfile(self.cfg_file):
            cfg = json.load(open(self.cfg_file))
        else:
            cfg = {}

        ### check config ###

        cfg['camera_list'] = [] if 'camera_list' not in cfg else cfg['camera_list']
        cfg['processor'] = () if 'processor' not in cfg else cfg['processor']
        cfg['dlc_cfg_files'] = [] if 'dlc_cfg_files' not in cfg else cfg['dlc_cfg_files']
        cfg['subjects'] = [] if 'subjects' not in cfg else cfg['subjects']
        cfg['directories'] = [] if 'directories' not in cfg else cfg['directories']

        return cfg


    def get_current_camera(self):
        if self.camera_name.get() == "Add Camera":
            return{'type' : "Add Camera"}
        else:
            cam_index = self.all_camera_names.index(self.camera_name.get())
            return self.cfg['camera_list'][cam_index]

    def set_current_camera(self, key, value):
        cam_index = self.all_camera_names.index(self.camera_name.get())
        self.cfg['camera_list'][cam_index][key] = value

    def set_exposure(self):
        # expos = float(self.exposure.get())
        # self.set_current_camera('exposure', expos)
        self.cam.set_exposure(float(self.exposure.get()))

    def set_crop(self):
        crop_vals = [int(v) for v in self.crop.get().strip().split(',')]
        crop_dict = {'top' : crop_vals[0], 'left' : crop_vals[1], 'height' : crop_vals[2], 'width' : crop_vals[3]}
        self.cam.set_crop(crop_vals[0], crop_vals[1], crop_vals[2], crop_vals[3])
        self.cam.close()
        self.cam = camera.ICCam(self.cam.serial_number, exposure=self.cam.get_exposure(), crop=crop_dict, rotate=int(self.rotate.get()))
        self.cam.open()

    def set_rotation(self):
        self.cam.set_rotation(int(self.rotate.get()))
        self.cam.close()
        crop_vals = [int(v) for v in self.crop.get().strip().split(',')]
        crop_dict = {'top' : crop_vals[0], 'left' : crop_vals[1], 'height' : crop_vals[2], 'width' : crop_vals[3]}
        self.cam = camera.ICCam(self.cam.serial_number, exposure=self.cam.get_exposure(), crop=crop_dict, rotate=int(self.rotate.get()))
        self.cam.open()

    def set_fps(self):
        fps = int(self.fps.get())
        # self.set_current_camera('fps', fps)
        self.cam.set_fps(int(self.fps.get()))

    def save_cam_settings(self):
        this_cam = self.get_current_camera()
        if this_cam['type'] != "VideoFeed":
            self.set_current_camera('exposure', float(self.exposure.get()))
            crop_vals = [int(v) for v in self.crop.get().strip().split(',')]
            crop_dict = {'top' : crop_vals[0], 'left' : crop_vals[1], 'height' : crop_vals[2], 'width' : crop_vals[3]}
            self.set_current_camera('crop', crop_dict)
            self.set_current_camera('rotate', int(self.rotate.get()))
            self.set_current_camera('fps', int(self.fps.get()))
        self.save_config(True)

    def browse_dlc_cfg(self):
        new_dlc_cfg = filedialog.askopenfilename()
        if new_dlc_cfg:
            self.dlc_cfg_file.set(new_dlc_cfg)
            ask_add_cfg = Tk()
            Label(ask_add_cfg, text="Would you like to add this cfg file to dropdown list?").pack()
            Button(ask_add_cfg, text="Yes", command=lambda: self.add_dlc_cfg(ask_add_cfg)).pack()
            Button(ask_add_cfg, text="No", command=ask_add_cfg.destroy).pack()

    def remove_dlc_cfg(self):
        self.cfg['dlc_cfg_files'].remove(self.dlc_cfg_file.get())
        self.dlc_cfg_file_entry['values'] = self.cfg['dlc_cfg_files']
        self.dlc_cfg_file.set('')
        self.save_config()

    def add_dlc_cfg(self, window):
        window.destroy()
        if self.dlc_cfg_file.get() not in self.cfg['dlc_cfg_files']:
            self.cfg['dlc_cfg_files'].append(self.dlc_cfg_file.get())
            self.dlc_cfg_file_entry['values'] = self.cfg['dlc_cfg_files']
            self.save_config()

    def add_subject(self):
        if self.subject.get():
            if self.subject.get() not in self.cfg['subjects']:
                self.cfg['subjects'].append(self.subject.get())
                self.subject_entry['values'] = self.cfg['subjects']
                self.save_config()

    def remove_subject(self):
        self.cfg['subjects'].remove(self.subject.get())
        self.subject_entry['values'] = self.cfg['subjects']
        self.save_config()
        self.subject.set('')

    def browse_directory(self):
        new_dir = tkfilebrowser.askopendirname()
        if new_dir:
            self.directory.set(new_dir)
            ask_add_dir = Tk()
            Label(ask_add_dir, text="Would you like to add this directory to dropdown list?").pack()
            Button(ask_add_dir, text="Yes", command=lambda: self.add_directory(ask_add_dir)).pack()
            Button(ask_add_dir, text="No", command=ask_add_dir.destroy).pack()

    def add_directory(self, window):
        window.destroy()
        if self.directory.get() not in self.cfg['directories']:
            self.cfg['directories'].append(self.directory.get())
            self.directory_entry['values'] = self.cfg['directories']
            self.save_config()

    def save_config(self, notify=False):
        json.dump(self.cfg, open(self.cfg_file, 'w'))
        if notify:
            messagebox.showinfo(title="Config file saved", message="Configuration file has been saved...")

    def remove_cam_cfg(self):
        if self.camera_name.get() != "Add Camera":
            delete = messagebox.askyesno(title="Delete Camera?", message="Are you sure you want to delete '%s'?" % self.camera_name.get())
            if delete:
                cam_index = self.all_camera_names.index(self.camera_name.get())
                self.cfg['camera_list'].pop(cam_index)
                self.get_camera_names()
                self.camera_entry['values'] = self.all_camera_names + ('Add Camera',)
                self.camera_entry.current(0)
                self.save_config(notify=True)

    def default_cam_specs(self, type="ICCam"):
        cam = {'name' : self.cam_name.get()}
        if type == "ICCam":
            cam['type'] = 'ICCam'
            cam['serial'] = self.cam_serial.get()
            cam['crop'] = {'top' : 0, 'left' : 0, 'height' : 540, 'width' : 720}
            cam['rotate'] = 0
            cam['exposure'] = .005
            cam['fps'] = 100
        elif type == "VideoFeed":
            cam['type'] = 'VideoFeed'
            cam['file'] = self.cam_file.get()
            cam['display'] = True if self.cam_display.get() == "True" else False
            cam['rotate'] = 0
            cam['exposure'] = 0
            cam['fps'] = 100
        return cam

    def add_cam_to_list(self, type, gui):
        new_cam = self.default_cam_specs(type)
        self.cfg['camera_list'].append(new_cam)
        self.get_camera_names()
        self.camera_entry['values'] = self.all_camera_names + ('Add Camera',)
        messagebox.showinfo("Camera Added", "Camera has been added to the dropdown menu.\nPlease update camera settings in the main window,\nand click 'Save Camera Settings' to save for future sessions.")
        gui.destroy()

    def update_camera_gui(self, gui, type, cur_row):

        if type == "ICCam":

            Label(gui, text="Serial: ").grid(sticky="w", row=cur_row, column=0)
            self.cam_serial = StringVar(gui)
            serial_entry = Combobox(gui, textvariable=self.cam_serial)
            serial_entry['values'] = tuple([s.decode() for s in camera.tisgrabber.TIS_CAM().GetDevices()])
            serial_entry.current(0)
            serial_entry.grid(sticky="nsew", row=cur_row, column=1)
            cur_row += 1

        elif type == "VideoFeed":

            Label(gui, text="File: ").grid(sticky="w", row=cur_row, column=0)
            self.cam_file = StringVar(gui, value="")
            Entry(gui, textvariable=self.cam_file).grid(sticky="nsew", row=cur_row, column=1)
            cur_row += 1

            Label(gui, text="Display: ").grid(sticky="w", row=cur_row, column=0)
            self.cam_display = StringVar(gui, value="True")
            Combobox(gui, textvariable=self.cam_display, values=("True", "False")).grid(sticky="nsew", row=cur_row, column=1)
            cur_row += 1

        Label(gui, text="Name: ").grid(sticky="w", row=cur_row, column=0)
        self.cam_name = StringVar(gui)
        Entry(gui, textvariable=self.cam_name).grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        Button(gui, text="Add Camera", command=lambda: self.add_cam_to_list(type, gui)).grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        Button(gui, text="Cancel", command=gui.destroy).grid(sticky="nsew", row=cur_row, column=1)

    def add_camera_gui(self):

        add_cam = Tk()
        cur_row = 0

        Label(add_cam, text="Type: ").grid(sticky="w", row=cur_row, column=0)
        self.cam_type = StringVar(add_cam)

        type_entry = Combobox(add_cam, textvariable=self.cam_type)
        type_entry['values'] = tuple([c[0] for c in inspect.getmembers(camera, inspect.isclass)])[1:]
        type_entry.current(0)
        type_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(add_cam, text="Select", command=lambda: self.update_camera_gui(add_cam, type_entry.get(), cur_row+1)).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        add_cam.mainloop()

    def init_cam(self):

        if self.cam:
            self.cam.close()

        this_cam = self.get_current_camera()

        if this_cam['type'] == "Add Camera":

            self.add_camera_gui()
            return

        else:

            setup_window = Tk()
            setup_window.title("Setting up camera...")
            Label(setup_window, text="Setting up camera, please wait...").pack()
            setup_window.update()

            if this_cam['type'] == 'ICCam':
                self.cam = camera.ICCam(this_cam['serial'], exposure=this_cam['exposure'], crop=this_cam['crop'], rotate=this_cam['rotate'])
            elif this_cam['type'] == 'VideoFeed':
                self.cam = camera.VideoFeed(this_cam['file'], display=this_cam['display'])

            if 'exposure' in this_cam:
                self.exposure.set(this_cam['exposure'])
            if 'crop' in this_cam:
                self.crop.set("%d, %d, %d, %d" % (this_cam['crop']['top'], this_cam['crop']['left'], this_cam['crop']['height'], this_cam['crop']['width']))
            if 'rotate' in this_cam:
                self.rotate.set(this_cam['rotate'])
            if 'fps' in this_cam:
                self.fps.set(this_cam['fps'])
            else:
                self.fps.set(self.cam.fps)

            self.cam.open()

            setup_window.destroy()

        self.cam.start_capture()

    def start_proc(self, gui):

        proc_param_dict = {}
        for i in range(1, len(self.proc_param_names)):
            proc_param_dict[self.proc_param_names[i]] = self.proc_param_default_types[i](self.proc_param_values[i-1].get())

        self.proc = self.proc_object(**proc_param_dict)
        gui.destroy()
        self.proc_button['text'] = 'Reset'

    def init_proc(self):

        ### load module ###
        self.proc_object = getattr(processor, self.proc_entry.get())
        args = inspect.getargspec(self.proc_object)
        self.proc_param_names = args[0]
        self.proc_param_default_values = args[3]
        self.proc_param_default_types = [type(v) for v in args[3]]
        for i in range(len(args[0])-len(args[3])):
            self.proc_param_default_values = ('',) + self.proc_param_default_values
            self.proc_param_default_types = [str] + self.proc_param_default_types

        proc_param_gui = Tk()
        self.proc_param_values = []
        for i in range(1,len(self.proc_param_names)):
            Label(proc_param_gui, text=self.proc_param_names[i]+": ").grid(sticky="w", row=i, column=0)
            self.proc_param_values.append(StringVar(proc_param_gui, value=str(self.proc_param_default_values[i])))
            Entry(proc_param_gui, textvariable=self.proc_param_values[i-1]).grid(sticky="nsew", row=i, column=1)

        Button(proc_param_gui, text="Init Proc", command=lambda: self.start_proc(proc_param_gui)).grid(sticky="nsew", row=i+1, column=1)

    def set_up_dlc(self):

        if self.dlc_live is not None:
            self.cam.stop_pose_estimation()

        dlc_setup = Tk()
        Label(dlc_setup, text="Initializing DLC Network, please wait...").pack()
        dlc_setup.update()

        iteration = None if self.iteration.get() == 'Default' else int(self.iteration.get())
        shuffle = int(self.shuffle.get())
        useFrozen = True if self.dlc_options.get() == 'Frozen' else False
        TFGPUinference = True if self.dlc_options.get() == 'TFGPUinference' else False

        self.dlc_live = DLCLive(self.dlc_cfg_file.get(), iteration=iteration, shuffle=shuffle, useFrozen=useFrozen, TFGPUinference=TFGPUinference)
        self.cam.start_pose_estimation(self.dlc_live)

        dlc_setup.destroy()
        self.dlc_button['text'] = 'Reset DLC Live'

    def set_up_session(self):

        ### check if video is currently open ###

        if self.cam.video_writer:
            vid_open_window = Tk()
            Label(vid_open_window, text="Video is currently open! \nPlease release the current video (click 'Save Video', even if no frames have been recorded) before setting up a new one.").pack()
            Button(vid_open_window, text="Ok", command=vid_open_window.destroy).pack()
            vid_open_window.mainloop()
            return

        ### check if camera is already set up ###

        if not self.cam:

            cam_check_window = Tk()
            Label(cam_check_window, text="No camera is found! \nPlease initialize camera before setting up video.").pack()
            Button(cam_check_window, text="Ok", command=lambda:cam_check_window.destroy).pack()
            cam_check_window.mainloop()

        else:

            month = datetime.datetime.now().month
            month = str(month) if month >= 10 else '0'+str(month)
            day = datetime.datetime.now().day
            day = str(day) if day >= 10 else '0'+str(day)
            year = str(datetime.datetime.now().year)
            date = year+'-'+month+'-'+day
            self.out_dir = self.directory.get()
            if not os.path.isdir(os.path.normpath(self.out_dir)):
                os.makedirs(os.path.normpath(self.out_dir))

            ### create output file names ###.

            self.base_name = self.get_current_camera()['name'].replace(" ", "") + '_' + self.subject.get() + '_' + date + '_' + self.attempt.get()
            self.vid_file = os.path.normpath(self.out_dir + '/VIDEO_' + self.base_name + '.avi')
            self.ts_file = os.path.normpath(self.out_dir + '/TIMESTAMPS_' + self.base_name + '.pickle')
            self.dlc_file = os.path.normpath(self.out_dir + '/DLC_' + self.base_name + '.h5')
            self.proc_file = os.path.normpath(self.out_dir + '/PROC_' + self.base_name + '.pickle')

            # check if files already exist
            if os.path.isfile(self.vid_file) or os.path.isfile(self.ts_file) or os.path.isfile(self.dlc_file) or os.path.isfile(self.proc_file):
                self.overwrite = False
                self.ask_overwrite = Tk()
                def quit_overwrite(ow):
                    self.overwrite = ow
                    self.ask_overwrite.quit()
                    self.ask_overwrite.destroy()
                Label(self.ask_overwrite, text="Files already exist with attempt number = %s. Would you like to overwrite the file?" % self.attempt.get()).pack()
                Button(self.ask_overwrite, text="Overwrite", command=lambda:quit_overwrite(True)).pack()
                Button(self.ask_overwrite, text="Cancel & pick new attempt number", command=lambda:quit_overwrite(False)).pack()
                self.ask_overwrite.mainloop()

                if not self.overwrite:
                    return

            ### print session label to GUI ###

            self.session_label['text'] = self.base_name

            ### set up videowriter ###

            self.cam.create_video_writer(self.vid_file)

    def start_record(self):
        record = False
        if self.cam:
            record = self.cam.start_record()
        if not record:
            no_record = Tk()
            Label(no_record, text="Either camera or video writer is not initialized.\nPlease set up camera and video before recording").pack()
            Button(no_record, text="Ok", command=no_record.destroy).pack()
            no_record.mainloop()
            self.record_on.set(0)

    def stop_record(self):
        if self.cam:
            self.cam.stop_record()

    def save_vid(self, delete=False):

        ### perform checks ###

        if not self.cam:
            messagebox.showwarning("No Camera", "Camera has not yet been initialized, no video recorded.")
            return

        elif not self.cam.video_writer:
            messagebox.showwarning("No Video Writer", "Video was not set up, no video recorded.")
            return

        elif delete:
            delete = messagebox.askokcancel("Delete Video?", "Do you wish to delete the video?")

        elif not self.cam.frame_times_record:
            messagebox.showwarning("No Frames Recorded", "No frames were recorded, video will be deleted")
            delete = True


        ### save or delete video ###

        if delete:

            self.cam.close_video()
            os.remove(self.vid_file)
            if os.path.isfile(self.ts_file):
                os.remove(self.ts_file)
            if os.path.isfile(self.dlc_file):
                os.remove(self.dlc_file)
            if os.path.isfile(self.proc_file):
                os.remove(self.proc_file)
            messagebox.showinfo("Video Deleted", "Video, timestamp, DLC, and processor files have been deleted.")

            # # are you sure GUI
            # def delete_video(window):
            #     window.destroy()
            #     self.cam.close_video()
            #     os.remove(self.vid_file)
            # confirm_delete = Tk()
            # Label(confirm_delete, text="Are you sure you want to delete the video file?").pack()
            # Button(confirm_delete, text="Yes", command=lambda: delete_video(confirm_delete)).pack()
            # Button(confirm_delete, text="No", command=confirm_delete.destroy).pack()
            # confirm_delete.mainloop()

        else:

            # save timestamps as pickle file. Includes frame_times, ttl_in_times, and ttl_out_times
            pickle.dump({'frame_times' : self.cam.frame_times_record}, open(self.ts_file, 'wb'))

            # TO DO convert poses to pandas df, save (with pose timestamps) to h5 file
            if self.cam.pose_times:
                pose_df = self.cam.get_pose_df()
                pose_df.to_hdf(self.dlc_file, key='df_with_missing', mode='w')
                # pose_df = self.cam.get_pose_df(self.dlc_live.cfg['bodyparts'])
                # pose_df.to_hdf(self.dlc_file, 'w')
                # pickle.dump({'poses' : self.cam.poses, 'frame_times' : self.cam.pose_frame_times_record, 'pose_times' : self.cam.pose_times}, open(self.dlc_file, 'wb'))

            if self.proc:
                self.proc.save(self.proc_file)

            # save video
            self.cam.close_video()
            messagebox.showinfo("Files Saved", "Files have been saved.")

        self.session_label['text'] = ""



    def closeGUI(self):
        if self.cam:
            self.cam.close()
        self.window.destroy()

    def createGUI(self):

        ### initialize window ###

        self.window = Tk()
        self.window.title("DeepLabCut Live")
        cur_row = 0


        ### select camera ###

        # camera entry
        Label(self.window, text="Camera "+": ").grid(sticky="w", row=cur_row, column=0)
        self.camera_name = StringVar(self.window)
        self.camera_entry = Combobox(self.window, textvariable=self.camera_name)
        self.camera_entry['values'] = self.all_camera_names + ('Add Camera',)
        self.camera_entry.current(0)
        self.camera_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Init Cam", command=self.init_cam).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # exposure
        Label(self.window, text="Exposure: ").grid(sticky="w", row=cur_row, column=0)
        self.exposure = StringVar()
        self.exposure_entry = Entry(self.window, textvariable=self.exposure)
        self.exposure_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Set Exposure", command=self.set_exposure).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # crop
        Label(self.window, text="Crop: ").grid(sticky="w", row=cur_row, column=0)
        self.crop = StringVar()
        self.crop_entry = Entry(self.window, textvariable=self.crop)
        self.crop_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Set Crop", command=self.set_crop).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # rotate
        Label(self.window, text="Rotate: ").grid(sticky="w", row=cur_row, column=0)
        self.rotate = StringVar()
        self.rotate_entry = Entry(self.window, textvariable=self.rotate)
        self.rotate_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Set Rotation", command=self.set_rotation).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # fps
        Label(self.window, text="FPS: ").grid(sticky="w", row=cur_row, column=0)
        self.fps = StringVar()
        self.fps_entry = Entry(self.window, textvariable=self.fps)
        self.fps_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Set FPS", command=self.set_fps).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # save camera settings and remove camera from config buttons
        Button(self.window, text="Save Camera Settings", command=self.save_cam_settings).grid(sticky="nsew", row=cur_row, column=1, columnspan=1)
        Button(self.window, text="Remove Camera", command=self.remove_cam_cfg).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # empty row
        Label(self.window, text="").grid(row=cur_row, column=0)
        cur_row += 1

        ### set up proc ###

        Label(self.window, text="Processor: ").grid(sticky='w', row=cur_row, column=0)
        self.proc_entry= StringVar()
        self.proc_entry = Combobox(self.window, textvariable=self.proc_entry)
        self.proc_entry['values'] = tuple([c[0] for c in inspect.getmembers(processor, inspect.isclass)])
        self.proc_entry.current(0)
        self.proc_entry.grid(sticky="nsew", row=cur_row, column=1)
        self.proc_button = Button(self.window, text="Set", command=self.init_proc)
        self.proc_button.grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # empty row
        Label(self.window, text='').grid(row=cur_row, column=0)
        cur_row += 1

        ### set up dlc live ###

        Label(self.window, text="DLC Config: ").grid(sticky='w', row=cur_row, column=0)
        self.dlc_cfg_file = StringVar()
        self.dlc_cfg_file_entry = Combobox(self.window, textvariable=self.dlc_cfg_file)
        if self.cfg['dlc_cfg_files']:
            self.dlc_cfg_file_entry['values'] = self.cfg['dlc_cfg_files']
            self.dlc_cfg_file_entry.current(0)
        self.dlc_cfg_file_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Browse", command=self.browse_dlc_cfg).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        Label(self.window, text="Iteration: ").grid(sticky='w', row=cur_row, column=0)
        self.iteration = StringVar()
        self.iteration_entry = Combobox(self.window, textvariable=self.iteration)
        self.iteration_entry['values'] = ("Default",) + tuple(range(10))
        self.iteration_entry.current(0)
        self.iteration_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Delete DLC Config", command=self.remove_dlc_cfg).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        Label(self.window, text="Shuffle: ").grid(sticky='w', row=cur_row, column=0)
        self.shuffle = StringVar()
        self.shuffle_entry = Combobox(self.window, textvariable=self.shuffle)
        self.shuffle_entry['values'] = tuple(range(1,10))
        self.shuffle_entry.current(0)
        self.shuffle_entry.grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        Label(self.window, text="Options: ").grid(sticky='w', row=cur_row, column=0)
        self.dlc_options = StringVar()
        self.dlc_options_entry = Combobox(self.window, textvariable=self.dlc_options)
        self.dlc_options_entry['values'] = ("Base", "TFGPUinference", "Frozen")
        self.dlc_options_entry.current(2)
        self.dlc_options_entry.grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        self.dlc_button = Button(self.window, text="Set DLC Live", command=self.set_up_dlc)
        self.dlc_button.grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        # empty row
        Label(self.window, text="").grid(row=cur_row, column=0)
        cur_row += 1

        ### set up session ###

        # subject
        Label(self.window, text="Subject: ").grid(sticky="w", row=cur_row, column=0)
        self.subject = StringVar()
        self.subject_entry = Combobox(self.window, textvariable=self.subject)
        self.subject_entry['values'] = self.cfg['subjects']
        self.subject_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Add Subject", command=self.add_subject).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # attempt
        Label(self.window, text="Attempt: ").grid(sticky="w", row=cur_row, column=0)
        self.attempt = StringVar()
        self.attempt_entry = Combobox(self.window, textvariable=self.attempt)
        self.attempt_entry['values'] = tuple(range(1,10))
        self.attempt_entry.current(0)
        self.attempt_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Remove Subject", command=self.remove_subject).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # out directory
        Label(self.window, text="Directory: ").grid(sticky="w", row=cur_row, column=0)
        self.directory = StringVar()
        self.directory_entry = Combobox(self.window, textvariable=self.directory)
        if self.cfg['directories']:
            self.directory_entry['values'] = self.cfg['directories']
            self.directory_entry.current(0)
        self.directory_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Browse", command=self.browse_directory).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        # set up session
        Button(self.window, text="Set Up Session", command=self.set_up_session).grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1
        Label(self.window, text="Current ::").grid(sticky="w", row=cur_row, column=0)
        self.session_label = Label(self.window, text="")
        self.session_label.grid(sticky="w", row=cur_row, column=1, columnspan=2)
        cur_row += 1

        # empty row
        Label(self.window, text="").grid(row=cur_row, column=0)
        cur_row += 1

        ### control recording ###

        Label(self.window, text="Record: ").grid(sticky="w",row=cur_row, column=0)
        self.record_on = IntVar(value=0)
        self.button_on = Radiobutton(self.window, text="On", selectcolor='green', indicatoron=0, variable=self.record_on, value=1, command=self.start_record).grid(sticky="nsew", row=cur_row, column=1)
        self.button_off = Radiobutton(self.window, text="Off", selectcolor='red', indicatoron=0, variable=self.record_on, value=0, command=self.stop_record).grid(sticky="nsew", row=cur_row+1, column=1)
        self.release_vid_save = Button(self.window, text="Save Video", command=lambda: self.save_vid()).grid(sticky="nsew", row=cur_row, column=2)
        self.release_vid_delete = Button(self.window, text="Delete Video", command=lambda: self.save_vid(delete=True)).grid(sticky="nsew", row=cur_row+1, column=2)
        cur_row += 2

        # empty row
        Label(self.window, text="").grid(row=cur_row, column=0)
        cur_row += 1

        ### close program ###

        Button(self.window, text="Close", command=self.closeGUI).grid(sticky="nsew", row=cur_row, column=0, columnspan=2)

    def runGUI(self):

        self.window.mainloop()


if __name__ == "__main__":
    dlc_live_gui = DLCLiveGUI()
    dlc_live_gui.runGUI()
