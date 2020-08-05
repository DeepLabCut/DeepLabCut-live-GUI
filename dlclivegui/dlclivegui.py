"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


from tkinter import (
    Tk,
    Toplevel,
    Label,
    Entry,
    Button,
    Radiobutton,
    Checkbutton,
    StringVar,
    IntVar,
    BooleanVar,
    filedialog,
    messagebox,
    simpledialog,
)
from tkinter.ttk import Combobox
import os
import sys
import glob
import json
import datetime
import inspect
import importlib

from PIL import Image, ImageTk, ImageDraw
import colorcet as cc

from dlclivegui import CameraPoseProcess
from dlclivegui import processor
from dlclivegui import camera
from dlclivegui.tkutil import SettingsWindow


class DLCLiveGUI(object):
    """ GUI to run DLC Live experiment
    """

    def __init__(self):
        """ Constructor method
        """

        ### check if documents path exists

        if not os.path.isdir(self.get_docs_path()):
            os.mkdir(self.get_docs_path())
        if not os.path.isdir(os.path.dirname(self.get_config_path(""))):
            os.mkdir(os.path.dirname(self.get_config_path("")))

        ### get configuration ###

        self.cfg_list = [
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.dirname(self.get_config_path("")) + "/*.json")
        ]

        ### initialize variables

        self.cam_pose_proc = None
        self.dlc_proc_params = None

        self.display_window = None
        self.display_cmap = None
        self.display_colors = None
        self.display_radius = None
        self.display_lik_thresh = None

        ### create GUI window ###

        self.createGUI()

    def get_docs_path(self):
        """ Get path to documents folder

        Returns
        -------
        str
            path to documents folder
        """

        return os.path.normpath(os.path.expanduser("~/Documents/DeepLabCut-live-GUI"))

    def get_config_path(self, cfg_name):
        """ Get path to configuration foler

        Parameters
        ----------
        cfg_name : str
            name of config file

        Returns
        -------
        str
            path to configuration file
        """

        return os.path.normpath(self.get_docs_path() + "/config/" + cfg_name + ".json")

    def get_config(self, cfg_name):
        """ Read configuration

        Parameters
        ----------
        cfg_name : str
            name of configuration
        """

        ### read configuration file ###

        self.cfg_file = self.get_config_path(cfg_name)
        if os.path.isfile(self.cfg_file):
            cfg = json.load(open(self.cfg_file))
        else:
            cfg = {}

        ### check config ###

        cfg["cameras"] = {} if "cameras" not in cfg else cfg["cameras"]
        cfg["processor_dir"] = (
            [] if "processor_dir" not in cfg else cfg["processor_dir"]
        )
        cfg["processor_args"] = (
            {} if "processor_args" not in cfg else cfg["processor_args"]
        )
        cfg["dlc_options"] = {} if "dlc_options" not in cfg else cfg["dlc_options"]
        cfg["dlc_display_options"] = (
            {} if "dlc_display_options" not in cfg else cfg["dlc_display_options"]
        )
        cfg["subjects"] = [] if "subjects" not in cfg else cfg["subjects"]
        cfg["directories"] = [] if "directories" not in cfg else cfg["directories"]

        self.cfg = cfg

    def change_config(self, event=None):
        """ Change configuration, update GUI menus

        Parameters
        ----------
        event : tkinter event, optional
            event , by default None
        """

        if self.cfg_name.get() == "Create New Config":
            new_name = simpledialog.askstring(
                "", "Please enter a name (no special characters).", parent=self.window
            )
            self.cfg_name.set(new_name)
        self.get_config(self.cfg_name.get())

        self.camera_entry["values"] = tuple(self.cfg["cameras"].keys()) + (
            "Add Camera",
        )
        self.camera_name.set("")
        self.dlc_proc_dir_entry["values"] = tuple(self.cfg["processor_dir"])
        self.dlc_proc_dir.set("")
        self.dlc_proc_name_entry["values"] = tuple()
        self.dlc_proc_name.set("")
        self.dlc_options_entry["values"] = tuple(self.cfg["dlc_options"].keys()) + (
            "Add DLC",
        )
        self.dlc_option.set("")
        self.subject_entry["values"] = tuple(self.cfg["subjects"])
        self.subject.set("")
        self.directory_entry["values"] = tuple(self.cfg["directories"])
        self.directory.set("")

    def remove_config(self):
        """ Remove configuration
        """

        cfg_name = self.cfg_name.get()
        delete_setup = messagebox.askyesnocancel(
            "Delete Config Permanently?",
            "Would you like to delete the configuration {} permanently (yes),\nremove the setup from the list for this session (no),\nor neither (cancel).".format(
                cfg_name
            ),
            parent=self.window,
        )
        if delete_setup is not None:
            if delete_setup:
                os.remove(self.get_config_path(cfg_name))
            self.cfg_list.remove(cfg_name)
            self.cfg_entry["values"] = tuple(self.cfg_list) + ("Create New Setup",)
            self.cfg_name.set("")

    def get_camera_names(self):
        """ Get camera names from configuration as a tuple
        """

        return tuple(self.cfg["cameras"].keys())

    def init_cam(self):
        """ Initialize camera
        """

        if self.cam_pose_proc is not None:
            messagebox.showerror(
                "Camera Exists",
                "Camera already exists! Please close current camera before initializing a new one.",
            )
            return

        this_cam = self.get_current_camera()

        if not this_cam:

            messagebox.showerror(
                "No Camera",
                "No camera selected. Please select a camera before initializing.",
                parent=self.window,
            )

        else:

            if this_cam["type"] == "Add Camera":

                self.add_camera_window()
                return

            else:

                self.cam_setup_window = Toplevel(self.window)
                self.cam_setup_window.title("Setting up camera...")
                Label(
                    self.cam_setup_window, text="Setting up camera, please wait..."
                ).pack()
                self.cam_setup_window.update()

                cam_obj = getattr(camera, this_cam["type"])
                cam = cam_obj(**this_cam["params"])
                self.cam_pose_proc = CameraPoseProcess(cam)
                ret = self.cam_pose_proc.start_capture_process()

                if cam.use_tk_display:
                    self.set_display_window()

                self.cam_setup_window.destroy()

    def get_current_camera(self):
        """ Get dictionary of the current camera
        """

        if self.camera_name.get():
            if self.camera_name.get() == "Add Camera":
                return {"type": "Add Camera"}
            else:
                return self.cfg["cameras"][self.camera_name.get()]

    def set_camera_param(self, key, value):
        """ Set a camera parameter
        """

        self.cfg["cameras"][self.camera_name.get()]["params"][key] = value

    def add_camera_window(self):
        """ Create gui to add a camera
        """

        add_cam = Tk()
        cur_row = 0

        Label(add_cam, text="Type: ").grid(sticky="w", row=cur_row, column=0)
        self.cam_type = StringVar(add_cam)

        cam_types = [c[0] for c in inspect.getmembers(camera, inspect.isclass)]
        cam_types = [c for c in cam_types if (c != "Camera") & ("Error" not in c)]

        type_entry = Combobox(add_cam, textvariable=self.cam_type, state="readonly")
        type_entry["values"] = tuple(cam_types)
        type_entry.current(0)
        type_entry.grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        Label(add_cam, text="Name: ").grid(sticky="w", row=cur_row, column=0)
        self.new_cam_name = StringVar(add_cam)
        Entry(add_cam, textvariable=self.new_cam_name).grid(
            sticky="nsew", row=cur_row, column=1
        )
        cur_row += 1

        Button(
            add_cam, text="Add Camera", command=lambda: self.add_cam_to_list(add_cam)
        ).grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        Button(add_cam, text="Cancel", command=add_cam.destroy).grid(
            sticky="nsew", row=cur_row, column=1
        )

        add_cam.mainloop()

    def add_cam_to_list(self, gui):
        """ Add new camera to the camera list
        """

        self.cfg["cameras"][self.new_cam_name.get()] = {
            "type": self.cam_type.get(),
            "params": {},
        }
        self.camera_name.set(self.new_cam_name.get())
        self.camera_entry["values"] = self.get_camera_names() + ("Add Camera",)
        self.save_config()
        # messagebox.showinfo("Camera Added", "Camera has been added to the dropdown menu. Please edit camera settings before initializing the new camera.", parent=gui)
        gui.destroy()

    def edit_cam_settings(self):
        """ GUI window to edit camera settings
        """

        arg_names, arg_vals, arg_dtypes, arg_restrict = self.get_cam_args()

        settings_window = Toplevel(self.window)
        settings_window.title("Camera Settings")
        cur_row = 0
        combobox_width = 15

        entry_vars = []
        for n, v in zip(arg_names, arg_vals):

            Label(settings_window, text=n + ": ").grid(row=cur_row, column=0)

            if type(v) is list:
                v = [str(x) if x is not None else "" for x in v]
                v = ", ".join(v)
            else:
                v = v if v is not None else ""
            entry_vars.append(StringVar(settings_window, value=str(v)))

            if n in arg_restrict.keys():
                restrict_vals = arg_restrict[n]
                if type(restrict_vals[0]) is list:
                    restrict_vals = [
                        ", ".join([str(i) for i in rv]) for rv in restrict_vals
                    ]
                Combobox(
                    settings_window,
                    textvariable=entry_vars[-1],
                    values=restrict_vals,
                    state="readonly",
                    width=combobox_width,
                ).grid(sticky="nsew", row=cur_row, column=1)
            else:
                Entry(settings_window, textvariable=entry_vars[-1]).grid(
                    sticky="nsew", row=cur_row, column=1
                )

            cur_row += 1

        cur_row += 1
        Button(
            settings_window,
            text="Update",
            command=lambda: self.update_camera_settings(
                arg_names, entry_vars, arg_dtypes, settings_window
            ),
        ).grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1
        Button(settings_window, text="Cancel", command=settings_window.destroy).grid(
            sticky="nsew", row=cur_row, column=1
        )

        _, row_count = settings_window.grid_size()
        for r in range(row_count):
            settings_window.grid_rowconfigure(r, minsize=20)

        settings_window.mainloop()

    def get_cam_args(self):
        """ Get arguments for the new camera
        """

        this_cam = self.get_current_camera()
        cam_obj = getattr(camera, this_cam["type"])
        arg_restrict = cam_obj.arg_restrictions()

        cam_args = inspect.getfullargspec(cam_obj)
        n_args = len(cam_args[0][1:])
        n_vals = len(cam_args[3])
        arg_names = []
        arg_vals = []
        arg_dtype = []
        for i in range(n_args):
            arg_names.append(cam_args[0][i + 1])

            if arg_names[i] in this_cam["params"].keys():
                val = this_cam["params"][arg_names[i]]
            else:
                val = None if i < n_args - n_vals else cam_args[3][n_vals - n_args + i]
            arg_vals.append(val)

            dt_val = val if i < n_args - n_vals else cam_args[3][n_vals - n_args + i]
            dt = type(dt_val) if type(dt_val) is not list else type(dt_val[0])
            arg_dtype.append(dt)

        return arg_names, arg_vals, arg_dtype, arg_restrict

    def update_camera_settings(self, names, entries, dtypes, gui):
        """ Update camera settings from values input in settings GUI
        """

        gui.destroy()

        for name, entry, dt in zip(names, entries, dtypes):
            val = entry.get()
            val = val.split(",")
            val = [v.strip() for v in val]
            try:
                if dt is bool:
                    val = [True if v == "True" else False for v in val]
                else:
                    val = [dt(v) if v else None for v in val]
            except TypeError:
                pass
            val = val if len(val) > 1 else val[0]
            self.set_camera_param(name, val)

        self.save_config()

    def set_display_window(self):
        """ Create a video display window
        """

        self.display_window = Toplevel(self.window)
        self.display_frame_label = Label(self.display_window)
        self.display_frame_label.pack()
        self.display_frame()

    def set_display_colors(self, bodyparts):
        """ Set colors for keypoints

        Parameters
        ----------
        bodyparts : int
            the number of keypoints
        """

        all_colors = getattr(cc, self.display_cmap)
        self.display_colors = all_colors[:: int(len(all_colors) / bodyparts)]

    def display_frame(self):
        """ Display a frame in display window
        """

        if self.cam_pose_proc and self.display_window:

            frame = self.cam_pose_proc.get_display_frame()

            if frame is not None:

                img = Image.fromarray(frame)
                if frame.ndim == 3:
                    b, g, r = img.split()
                    img = Image.merge("RGB", (r, g, b))

                pose = (
                    self.cam_pose_proc.get_display_pose()
                    if self.display_keypoints.get()
                    else None
                )

                if pose is not None:

                    im_size = (frame.shape[1], frame.shape[0])

                    if not self.display_colors:
                        self.set_display_colors(pose.shape[0])

                    img_draw = ImageDraw.Draw(img)

                    for i in range(pose.shape[0]):
                        if pose[i, 2] > self.display_lik_thresh:
                            try:
                                x0 = (
                                    pose[i, 0] - self.display_radius
                                    if pose[i, 0] - self.display_radius > 0
                                    else 0
                                )
                                x1 = (
                                    pose[i, 0] + self.display_radius
                                    if pose[i, 0] + self.display_radius < im_size[1]
                                    else im_size[1]
                                )
                                y0 = (
                                    pose[i, 1] - self.display_radius
                                    if pose[i, 1] - self.display_radius > 0
                                    else 0
                                )
                                y1 = (
                                    pose[i, 1] + self.display_radius
                                    if pose[i, 1] + self.display_radius < im_size[0]
                                    else im_size[0]
                                )
                                coords = [x0, y0, x1, y1]
                                img_draw.ellipse(
                                    coords,
                                    fill=self.display_colors[i],
                                    outline=self.display_colors[i],
                                )
                            except Exception as e:
                                print(e)

                imgtk = ImageTk.PhotoImage(image=img)
                self.display_frame_label.imgtk = imgtk
                self.display_frame_label.configure(image=imgtk)

            self.display_frame_label.after(10, self.display_frame)

    def change_display_keypoints(self):
        """ Toggle display keypoints. If turning on, set display options. If turning off, destroy display window
        """

        if self.display_keypoints.get():

            display_options = self.cfg["dlc_display_options"][
                self.dlc_option.get()
            ].copy()
            self.display_cmap = display_options["cmap"]
            self.display_radius = display_options["radius"]
            self.display_lik_thresh = display_options["lik_thresh"]

            if not self.display_window:
                self.set_display_window()

        else:

            if self.cam_pose_proc is not None:
                if not self.cam_pose_proc.device.use_tk_display:
                    if self.display_window:
                        self.display_window.destroy()
                        self.display_window = None
                        self.display_colors = None

    def edit_dlc_display(self):

        display_options = self.cfg["dlc_display_options"][self.dlc_option.get()]

        dlc_display_settings = {
            "color map": {
                "value": display_options["cmap"],
                "dtype": str,
                "restriction": ["bgy", "kbc", "bmw", "bmy", "kgy", "fire"],
            },
            "radius": {"value": display_options["radius"], "dtype": int},
            "likelihood threshold": {
                "value": display_options["lik_thresh"],
                "dtype": float,
            },
        }

        dlc_display_gui = SettingsWindow(
            title="Edit DLC Display Settings",
            settings=dlc_display_settings,
            parent=self.window,
        )

        dlc_display_gui.mainloop()
        display_settings = dlc_display_gui.get_values()

        display_options["cmap"] = display_settings["color map"]
        display_options["radius"] = display_settings["radius"]
        display_options["lik_thresh"] = display_settings["likelihood threshold"]

        self.display_cmap = display_options["cmap"]
        self.display_radius = display_options["radius"]
        self.display_lik_thresh = display_options["lik_thresh"]

        self.cfg["dlc_display_options"][self.dlc_option.get()] = display_options
        self.save_config()

    def close_camera(self):
        """ Close capture process and display
        """

        if self.cam_pose_proc:
            if self.display_window is not None:
                self.display_window.destroy()
                self.display_window = None
            ret = self.cam_pose_proc.stop_capture_process()

        self.cam_pose_proc = None

    def change_dlc_option(self, event=None):

        if self.dlc_option.get() == "Add DLC":
            self.edit_dlc_settings(True)

    def edit_dlc_settings(self, new=False):

        if new:
            cur_set = self.empty_dlc_settings()
        else:
            cur_set = self.cfg["dlc_options"][self.dlc_option.get()].copy()
            cur_set["name"] = self.dlc_option.get()
            cur_set["cropping"] = (
                ", ".join([str(c) for c in cur_set["cropping"]])
                if cur_set["cropping"]
                else ""
            )
            cur_set["dynamic"] = ", ".join([str(d) for d in cur_set["dynamic"]])
            cur_set["mode"] = (
                "Optimize Latency" if "mode" not in cur_set else cur_set["mode"]
            )

        self.dlc_settings_window = Toplevel(self.window)
        self.dlc_settings_window.title("DLC Settings")
        cur_row = 0

        Label(self.dlc_settings_window, text="Name: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_settings_name = StringVar(
            self.dlc_settings_window, value=cur_set["name"]
        )
        Entry(self.dlc_settings_window, textvariable=self.dlc_settings_name).grid(
            sticky="nsew", row=cur_row, column=1
        )
        cur_row += 1

        Label(self.dlc_settings_window, text="Model Path: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_settings_model_path = StringVar(
            self.dlc_settings_window, value=cur_set["model_path"]
        )
        Entry(self.dlc_settings_window, textvariable=self.dlc_settings_model_path).grid(
            sticky="nsew", row=cur_row, column=1
        )
        Button(
            self.dlc_settings_window, text="Browse", command=self.browse_dlc_path
        ).grid(sticky="nsew", row=cur_row, column=2)
        cur_row += 1

        Label(self.dlc_settings_window, text="Model Type: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_settings_model_type = StringVar(
            self.dlc_settings_window, value=cur_set["model_type"]
        )
        Combobox(
            self.dlc_settings_window,
            textvariable=self.dlc_settings_model_type,
            value=["base", "tensorrt", "tflite"],
            state="readonly",
        ).grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        Label(self.dlc_settings_window, text="Precision: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_settings_precision = StringVar(
            self.dlc_settings_window, value=cur_set["precision"]
        )
        Combobox(
            self.dlc_settings_window,
            textvariable=self.dlc_settings_precision,
            value=["FP32", "FP16", "INT8"],
            state="readonly",
        ).grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        Label(self.dlc_settings_window, text="Cropping: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_settings_cropping = StringVar(
            self.dlc_settings_window, value=cur_set["cropping"]
        )
        Entry(self.dlc_settings_window, textvariable=self.dlc_settings_cropping).grid(
            sticky="nsew", row=cur_row, column=1
        )
        cur_row += 1

        Label(self.dlc_settings_window, text="Dynamic: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_settings_dynamic = StringVar(
            self.dlc_settings_window, value=cur_set["dynamic"]
        )
        Entry(self.dlc_settings_window, textvariable=self.dlc_settings_dynamic).grid(
            sticky="nsew", row=cur_row, column=1
        )
        cur_row += 1

        Label(self.dlc_settings_window, text="Resize: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_settings_resize = StringVar(
            self.dlc_settings_window, value=cur_set["resize"]
        )
        Entry(self.dlc_settings_window, textvariable=self.dlc_settings_resize).grid(
            sticky="nsew", row=cur_row, column=1
        )
        cur_row += 1

        Label(self.dlc_settings_window, text="Mode: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_settings_mode = StringVar(
            self.dlc_settings_window, value=cur_set["mode"]
        )
        Combobox(
            self.dlc_settings_window,
            textvariable=self.dlc_settings_mode,
            state="readonly",
            values=["Optimize Latency", "Optimize Rate"],
        ).grid(sticky="nsew", row=cur_row, column=1)
        cur_row += 1

        Button(
            self.dlc_settings_window, text="Update", command=self.update_dlc_settings
        ).grid(sticky="nsew", row=cur_row, column=1)
        Button(
            self.dlc_settings_window,
            text="Cancel",
            command=self.dlc_settings_window.destroy,
        ).grid(sticky="nsew", row=cur_row, column=2)

    def empty_dlc_settings(self):

        return {
            "name": "",
            "model_path": "",
            "model_type": "base",
            "precision": "FP32",
            "cropping": "",
            "dynamic": "False, 0.5, 10",
            "resize": "1.0",
            "mode": "Optimize Latency",
        }

    def browse_dlc_path(self):
        """ Open file browser to select DLC exported model directory
        """

        new_dlc_path = filedialog.askdirectory(parent=self.dlc_settings_window)
        if new_dlc_path:
            self.dlc_settings_model_path.set(new_dlc_path)

    def update_dlc_settings(self):
        """ Update DLC settings for the current dlc option from DLC Settings GUI
        """

        precision = (
            self.dlc_settings_precision.get()
            if self.dlc_settings_precision.get()
            else "FP32"
        )

        crop_warn = False
        dlc_crop = self.dlc_settings_cropping.get()
        if dlc_crop:
            try:
                dlc_crop = dlc_crop.split(",")
                assert len(dlc_crop) == 4
                dlc_crop = [int(c) for c in dlc_crop]
            except Exception:
                crop_warn = True
                dlc_crop = None
        else:
            dlc_crop = None

        try:
            dlc_dynamic = self.dlc_settings_dynamic.get().replace(" ", "")
            dlc_dynamic = dlc_dynamic.split(",")
            dlc_dynamic[0] = True if dlc_dynamic[0] == "True" else False
            dlc_dynamic[1] = float(dlc_dynamic[1])
            dlc_dynamic[2] = int(dlc_dynamic[2])
            dlc_dynamic = tuple(dlc_dynamic)
            dyn_warn = False
        except Exception:
            dyn_warn = True
            dlc_dynamic = (False, 0.5, 10)

        dlc_resize = (
            float(self.dlc_settings_resize.get())
            if self.dlc_settings_resize.get()
            else None
        )
        dlc_mode = self.dlc_settings_mode.get()

        warn_msg = ""
        if crop_warn:
            warn_msg += "DLC Cropping was not set properly. Using default cropping parameters...\n"
        if dyn_warn:
            warn_msg += "DLC Dynamic Cropping was not set properly. Using default dynamic cropping parameters..."
        if warn_msg:
            messagebox.showerror(
                "DLC Settings Error", warn_msg, parent=self.dlc_settings_window
            )

        self.cfg["dlc_options"][self.dlc_settings_name.get()] = {
            "model_path": self.dlc_settings_model_path.get(),
            "model_type": self.dlc_settings_model_type.get(),
            "precision": precision,
            "cropping": dlc_crop,
            "dynamic": dlc_dynamic,
            "resize": dlc_resize,
            "mode": dlc_mode,
        }

        if self.dlc_settings_name.get() not in self.cfg["dlc_display_options"]:
            self.cfg["dlc_display_options"][self.dlc_settings_name.get()] = {
                "cmap": "bgy",
                "radius": 3,
                "lik_thresh": 0.5,
            }

        self.save_config()
        self.dlc_options_entry["values"] = tuple(self.cfg["dlc_options"].keys()) + (
            "Add DLC",
        )
        self.dlc_option.set(self.dlc_settings_name.get())
        self.dlc_settings_window.destroy()

    def remove_dlc_option(self):
        """ Delete DLC Option from config
        """

        del self.cfg["dlc_options"][self.dlc_option.get()]
        del self.cfg["dlc_display_options"][self.dlc_option.get()]
        self.dlc_options_entry["values"] = tuple(self.cfg["dlc_options"].keys()) + (
            "Add DLC",
        )
        self.dlc_option.set("")
        self.save_config()

    def init_dlc(self):
        """ Initialize DLC Live object
        """

        self.stop_pose()

        self.dlc_setup_window = Toplevel(self.window)
        self.dlc_setup_window.title("Setting up DLC...")
        Label(self.dlc_setup_window, text="Setting up DLC, please wait...").pack()
        self.dlc_setup_window.after(10, self.start_pose)
        self.dlc_setup_window.mainloop()

    def start_pose(self):

        dlc_params = self.cfg["dlc_options"][self.dlc_option.get()].copy()
        dlc_params["processor"] = self.dlc_proc_params
        ret = self.cam_pose_proc.start_pose_process(dlc_params)
        self.dlc_setup_window.destroy()

    def stop_pose(self):
        """ Stop pose process
        """

        if self.cam_pose_proc:
            ret = self.cam_pose_proc.stop_pose_process()

    def add_subject(self):
        new_sub = self.subject.get()
        if new_sub:
            if new_sub not in self.cfg["subjects"]:
                self.cfg["subjects"].append(new_sub)
                self.subject_entry["values"] = tuple(self.cfg["subjects"])
                self.save_config()

    def remove_subject(self):

        self.cfg["subjects"].remove(self.subject.get())
        self.subject_entry["values"] = self.cfg["subjects"]
        self.save_config()
        self.subject.set("")

    def browse_directory(self):

        new_dir = filedialog.askdirectory(parent=self.window)
        if new_dir:
            self.directory.set(new_dir)
            ask_add_dir = Tk()
            Label(
                ask_add_dir,
                text="Would you like to add this directory to dropdown list?",
            ).pack()
            Button(
                ask_add_dir, text="Yes", command=lambda: self.add_directory(ask_add_dir)
            ).pack()
            Button(ask_add_dir, text="No", command=ask_add_dir.destroy).pack()

    def add_directory(self, window):

        window.destroy()
        if self.directory.get() not in self.cfg["directories"]:
            self.cfg["directories"].append(self.directory.get())
            self.directory_entry["values"] = self.cfg["directories"]
            self.save_config()

    def save_config(self, notify=False):

        json.dump(self.cfg, open(self.cfg_file, "w"))
        if notify:
            messagebox.showinfo(
                title="Config file saved",
                message="Configuration file has been saved...",
                parent=self.window,
            )

    def remove_cam_cfg(self):

        if self.camera_name.get() != "Add Camera":
            delete = messagebox.askyesno(
                title="Delete Camera?",
                message="Are you sure you want to delete '%s'?"
                % self.camera_name.get(),
                parent=self.window,
            )
            if delete:
                del self.cfg["cameras"][self.camera_name.get()]
                self.camera_entry["values"] = self.get_camera_names() + ("Add Camera",)
                self.camera_name.set("")
                self.save_config()

    def browse_dlc_processor(self):

        new_dir = filedialog.askdirectory(parent=self.window)
        if new_dir:
            self.dlc_proc_dir.set(new_dir)
            self.update_dlc_proc_list()

            if new_dir not in self.cfg["processor_dir"]:
                if messagebox.askyesno(
                    "Add to dropdown",
                    "Would you like to add this directory to dropdown list?",
                ):
                    self.cfg["processor_dir"].append(new_dir)
                    self.dlc_proc_dir_entry["values"] = tuple(self.cfg["processor_dir"])
                    self.save_config()

    def rem_dlc_proc_dir(self):

        if self.dlc_proc_dir.get() in self.cfg["processor_dir"]:
            self.cfg["processor_dir"].remove(self.dlc_proc_dir.get())
            self.save_config()
        self.dlc_proc_dir_entry["values"] = tuple(self.cfg["processor_dir"])
        self.dlc_proc_dir.set("")

    def update_dlc_proc_list(self, event=None):

        ### if dlc proc module already initialized, delete module and remove from path ###

        self.processor_list = []

        if self.dlc_proc_dir.get():

            if hasattr(self, "dlc_proc_module"):
                sys.path.remove(sys.path[0])

            new_path = os.path.normpath(os.path.dirname(self.dlc_proc_dir.get()))
            if new_path not in sys.path:
                sys.path.insert(0, new_path)

            new_mod = os.path.basename(self.dlc_proc_dir.get())
            if new_mod in sys.modules:
                del sys.modules[new_mod]

            ### load new module ###

            processor_spec = importlib.util.find_spec(
                os.path.basename(self.dlc_proc_dir.get())
            )
            try:
                self.dlc_proc_module = importlib.util.module_from_spec(processor_spec)
                processor_spec.loader.exec_module(self.dlc_proc_module)
                # self.processor_list = inspect.getmembers(self.dlc_proc_module, inspect.isclass)
                self.processor_list = [
                    proc for proc in dir(self.dlc_proc_module) if "__" not in proc
                ]
            except AttributeError:
                if hasattr(self, "window"):
                    messagebox.showerror(
                        "Failed to load processors!",
                        "Failed to load processors from directory = "
                        + self.dlc_proc_dir.get()
                        + ".\nPlease select a different directory.",
                        parent=self.window,
                    )

            self.dlc_proc_name_entry["values"] = tuple(self.processor_list)

    def set_proc(self):

        # proc_param_dict = {}
        # for i in range(1, len(self.proc_param_names)):
        #     proc_param_dict[self.proc_param_names[i]] = self.proc_param_default_types[i](self.proc_param_values[i-1].get())

        # if self.dlc_proc_dir.get() not in self.cfg['processor_args']:
        #     self.cfg['processor_args'][self.dlc_proc_dir.get()] = {}
        # self.cfg['processor_args'][self.dlc_proc_dir.get()][self.dlc_proc_name.get()] = proc_param_dict
        # self.save_config()

        # self.dlc_proc = self.proc_object(**proc_param_dict)
        proc_object = getattr(self.dlc_proc_module, self.dlc_proc_name.get())
        self.dlc_proc_params = {"object": proc_object}
        self.dlc_proc_params.update(
            self.cfg["processor_args"][self.dlc_proc_dir.get()][
                self.dlc_proc_name.get()
            ]
        )

    def clear_proc(self):

        self.dlc_proc_params = None

    def edit_proc(self):

        ### get default args: load module and read arguments ###

        self.proc_object = getattr(self.dlc_proc_module, self.dlc_proc_name.get())
        def_args = inspect.getargspec(self.proc_object)
        self.proc_param_names = def_args[0]
        self.proc_param_default_values = def_args[3]
        self.proc_param_default_types = [
            type(v) if type(v) is not list else [type(v[0])] for v in def_args[3]
        ]
        for i in range(len(def_args[0]) - len(def_args[3])):
            self.proc_param_default_values = ("",) + self.proc_param_default_values
            self.proc_param_default_types = [str] + self.proc_param_default_types

        ### check for existing settings in config ###

        old_args = {}
        if self.dlc_proc_dir.get() in self.cfg["processor_args"]:
            if (
                self.dlc_proc_name.get()
                in self.cfg["processor_args"][self.dlc_proc_dir.get()]
            ):
                old_args = self.cfg["processor_args"][self.dlc_proc_dir.get()][
                    self.dlc_proc_name.get()
                ].copy()
        else:
            self.cfg["processor_args"][self.dlc_proc_dir.get()] = {}

        ### get dictionary of arguments ###

        proc_args_dict = {}
        for i in range(1, len(self.proc_param_names)):

            if self.proc_param_names[i] in old_args:
                this_value = old_args[self.proc_param_names[i]]
            else:
                this_value = self.proc_param_default_values[i]

            proc_args_dict[self.proc_param_names[i]] = {
                "value": this_value,
                "dtype": self.proc_param_default_types[i],
            }

        proc_args_gui = SettingsWindow(
            title="DLC Processor Settings", settings=proc_args_dict, parent=self.window
        )
        proc_args_gui.mainloop()

        self.cfg["processor_args"][self.dlc_proc_dir.get()][
            self.dlc_proc_name.get()
        ] = proc_args_gui.get_values()
        self.save_config()

    def init_session(self):

        ### check if video is currently open ###

        if self.record_on.get() > -1:
            messagebox.showerror(
                "Session Open",
                "Session is currently open! Please release the current video (click 'Save Video' of 'Delete Video', even if no frames have been recorded) before setting up a new one.",
                parent=self.window,
            )
            return

        ### check if camera is already set up ###

        if not self.cam_pose_proc:
            messagebox.showerror(
                "No Camera",
                "No camera is found! Please initialize a camera before setting up the video.",
                parent=self.window,
            )
            return

        ### set up session window

        self.session_setup_window = Toplevel(self.window)
        self.session_setup_window.title("Setting up session...")
        Label(
            self.session_setup_window, text="Setting up session, please wait..."
        ).pack()
        self.session_setup_window.after(10, self.start_writer)
        self.session_setup_window.mainloop()

    def start_writer(self):

        ### set up file name (get date and create directory)

        dt = datetime.datetime.now()
        date = f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
        self.out_dir = self.directory.get()
        if not os.path.isdir(os.path.normpath(self.out_dir)):
            os.makedirs(os.path.normpath(self.out_dir))

        ### create output file names

        self.base_name = os.path.normpath(
            f"{self.out_dir}/{self.camera_name.get().replace(' ', '')}_{self.subject.get()}_{date}_{self.attempt.get()}"
        )
        # self.vid_file = os.path.normpath(self.out_dir + '/VIDEO_' + self.base_name + '.avi')
        # self.ts_file = os.path.normpath(self.out_dir + '/TIMESTAMPS_' + self.base_name + '.pickle')
        # self.dlc_file = os.path.normpath(self.out_dir + '/DLC_' + self.base_name + '.h5')
        # self.proc_file = os.path.normpath(self.out_dir + '/PROC_' + self.base_name + '.pickle')

        ### check if files already exist

        fs = glob.glob(f"{self.base_name}*")
        if len(fs) > 0:
            overwrite = messagebox.askyesno(
                "Files Exist",
                "Files already exist with attempt number = {}. Would you like to overwrite the file?".format(
                    self.attempt.get()
                ),
                parent=self.session_setup_window,
            )
            if not overwrite:
                return

        ### start writer

        ret = self.cam_pose_proc.start_writer_process(self.base_name)

        self.session_setup_window.destroy()

        ### set GUI to Ready

        self.record_on.set(0)

    def start_record(self):
        """ Issues command to start recording frames and poses
        """

        ret = False
        if self.cam_pose_proc is not None:
            ret = self.cam_pose_proc.start_record()

        if not ret:
            messagebox.showerror(
                "Recording Not Ready",
                "Recording has not been set up. Please make sure a camera and session have been initialized.",
                parent=self.window,
            )
            self.record_on.set(-1)

    def stop_record(self):
        """ Issues command to stop recording frames and poses
        """

        if self.cam_pose_proc is not None:
            ret = self.cam_pose_proc.stop_record()
            self.record_on.set(0)

    def save_vid(self, delete=False):
        """ Saves video, timestamp, and DLC files

        Parameters
        ----------
        delete : bool, optional
            flag to delete created files, by default False
        """

        ### perform checks ###

        if self.cam_pose_proc is None:
            messagebox.showwarning(
                "No Camera",
                "Camera has not yet been initialized, no video recorded.",
                parent=self.window,
            )
            return

        elif self.record_on.get() == -1:
            messagebox.showwarning(
                "No Video File",
                "Video was not set up, no video recorded.",
                parent=self.window,
            )
            return

        elif self.record_on.get() == 1:
            messagebox.showwarning(
                "Active Recording",
                "You are currently recording a video. Please stop the video before saving.",
                parent=self.window,
            )
            return

        elif delete:
            delete = messagebox.askokcancel(
                "Delete Video?", "Do you wish to delete the video?", parent=self.window
            )

        ### save or delete video ###

        if delete:
            ret = self.cam_pose_proc.stop_writer_process(save=False)
            messagebox.showinfo(
                "Video Deleted",
                "Video and timestamp files have been deleted.",
                parent=self.window,
            )
        else:
            ret = self.cam_pose_proc.stop_writer_process(save=True)
            ret_pose = self.cam_pose_proc.save_pose(self.base_name)
            if ret:
                if ret_pose:
                    messagebox.showinfo(
                        "Files Saved",
                        "Video, timestamp, and DLC Files have been saved.",
                    )
                else:
                    messagebox.showinfo(
                        "Files Saved", "Video and timestamp files have been saved."
                    )
            else:
                messagebox.showwarning(
                    "No Frames Recorded",
                    "No frames were recorded, video was deleted",
                    parent=self.window,
                )

        self.record_on.set(-1)

    def closeGUI(self):

        if self.cam_pose_proc:
            ret = self.cam_pose_proc.stop_writer_process()
            ret = self.cam_pose_proc.stop_pose_process()
            ret = self.cam_pose_proc.stop_capture_process()

        self.window.destroy()

    def createGUI(self):

        ### initialize window ###

        self.window = Tk()
        self.window.title("DeepLabCut Live")
        cur_row = 0
        combobox_width = 15

        ### select cfg file
        if len(self.cfg_list) > 0:
            initial_cfg = self.cfg_list[0]
        else:
            initial_cfg = ""

        Label(self.window, text="Config: ").grid(sticky="w", row=cur_row, column=0)
        self.cfg_name = StringVar(self.window, value=initial_cfg)
        self.cfg_entry = Combobox(
            self.window, textvariable=self.cfg_name, width=combobox_width
        )
        self.cfg_entry["values"] = tuple(self.cfg_list) + ("Create New Config",)
        self.cfg_entry.bind("<<ComboboxSelected>>", self.change_config)
        self.cfg_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Remove Config", command=self.remove_config).grid(
            sticky="nsew", row=cur_row, column=2
        )

        self.get_config(initial_cfg)

        cur_row += 2

        ### select camera ###

        # camera entry
        Label(self.window, text="Camera: ").grid(sticky="w", row=cur_row, column=0)
        self.camera_name = StringVar(self.window)
        self.camera_entry = Combobox(self.window, textvariable=self.camera_name)
        cam_names = self.get_camera_names()
        self.camera_entry["values"] = cam_names + ("Add Camera",)
        if cam_names:
            self.camera_entry.current(0)
        self.camera_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Init Cam", command=self.init_cam).grid(
            sticky="nsew", row=cur_row, column=2
        )
        cur_row += 1

        Button(
            self.window, text="Edit Camera Settings", command=self.edit_cam_settings
        ).grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Close Camera", command=self.close_camera).grid(
            sticky="nsew", row=cur_row, column=2
        )
        cur_row += 1
        Button(self.window, text="Remove Camera", command=self.remove_cam_cfg).grid(
            sticky="nsew", row=cur_row, column=2
        )

        cur_row += 2

        ### set up proc ###

        Label(self.window, text="Processor Dir: ").grid(
            sticky="w", row=cur_row, column=0
        )
        self.dlc_proc_dir = StringVar(self.window)
        self.dlc_proc_dir_entry = Combobox(self.window, textvariable=self.dlc_proc_dir)
        self.dlc_proc_dir_entry["values"] = tuple(self.cfg["processor_dir"])
        if len(self.cfg["processor_dir"]) > 0:
            self.dlc_proc_dir_entry.current(0)
        self.dlc_proc_dir_entry.bind("<<ComboboxSelected>>", self.update_dlc_proc_list)
        self.dlc_proc_dir_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Browse", command=self.browse_dlc_processor).grid(
            sticky="nsew", row=cur_row, column=2
        )
        Button(self.window, text="Remove Proc Dir", command=self.rem_dlc_proc_dir).grid(
            sticky="nsew", row=cur_row + 1, column=2
        )
        cur_row += 2

        Label(self.window, text="Processor: ").grid(sticky="w", row=cur_row, column=0)
        self.dlc_proc_name = StringVar(self.window)
        self.dlc_proc_name_entry = Combobox(
            self.window, textvariable=self.dlc_proc_name
        )
        self.update_dlc_proc_list()
        # self.dlc_proc_name_entry['values'] = tuple(self.processor_list) # tuple([c[0] for c in inspect.getmembers(processor, inspect.isclass)])
        if len(self.processor_list) > 0:
            self.dlc_proc_name_entry.current(0)
        self.dlc_proc_name_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Set Proc", command=self.set_proc).grid(
            sticky="nsew", row=cur_row, column=2
        )
        Button(self.window, text="Edit Proc Settings", command=self.edit_proc).grid(
            sticky="nsew", row=cur_row + 1, column=1
        )
        Button(self.window, text="Clear Proc", command=self.clear_proc).grid(
            sticky="nsew", row=cur_row + 1, column=2
        )

        cur_row += 3

        ### set up dlc live ###

        Label(self.window, text="DeepLabCut: ").grid(sticky="w", row=cur_row, column=0)
        self.dlc_option = StringVar(self.window)
        self.dlc_options_entry = Combobox(self.window, textvariable=self.dlc_option)
        self.dlc_options_entry["values"] = tuple(self.cfg["dlc_options"].keys()) + (
            "Add DLC",
        )
        self.dlc_options_entry.bind("<<ComboboxSelected>>", self.change_dlc_option)
        self.dlc_options_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Init DLC", command=self.init_dlc).grid(
            sticky="nsew", row=cur_row, column=2
        )
        cur_row += 1

        Button(
            self.window, text="Edit DLC Settings", command=self.edit_dlc_settings
        ).grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Stop DLC", command=self.stop_pose).grid(
            sticky="nsew", row=cur_row, column=2
        )
        cur_row += 1

        self.display_keypoints = BooleanVar(self.window, value=False)
        Checkbutton(
            self.window,
            text="Display DLC Keypoints",
            variable=self.display_keypoints,
            indicatoron=0,
            command=self.change_display_keypoints,
        ).grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Remove DLC", command=self.remove_dlc_option).grid(
            sticky="nsew", row=cur_row, column=2
        )
        cur_row += 1

        Button(
            self.window, text="Edit DLC Display Settings", command=self.edit_dlc_display
        ).grid(sticky="nsew", row=cur_row, column=1)

        cur_row += 2

        ### set up session ###

        # subject
        Label(self.window, text="Subject: ").grid(sticky="w", row=cur_row, column=0)
        self.subject = StringVar(self.window)
        self.subject_entry = Combobox(self.window, textvariable=self.subject)
        self.subject_entry["values"] = self.cfg["subjects"]
        self.subject_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Add Subject", command=self.add_subject).grid(
            sticky="nsew", row=cur_row, column=2
        )
        cur_row += 1

        # attempt
        Label(self.window, text="Attempt: ").grid(sticky="w", row=cur_row, column=0)
        self.attempt = StringVar(self.window)
        self.attempt_entry = Combobox(self.window, textvariable=self.attempt)
        self.attempt_entry["values"] = tuple(range(1, 10))
        self.attempt_entry.current(0)
        self.attempt_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Remove Subject", command=self.remove_subject).grid(
            sticky="nsew", row=cur_row, column=2
        )
        cur_row += 1

        # out directory
        Label(self.window, text="Directory: ").grid(sticky="w", row=cur_row, column=0)
        self.directory = StringVar(self.window)
        self.directory_entry = Combobox(self.window, textvariable=self.directory)
        if self.cfg["directories"]:
            self.directory_entry["values"] = self.cfg["directories"]
            self.directory_entry.current(0)
        self.directory_entry.grid(sticky="nsew", row=cur_row, column=1)
        Button(self.window, text="Browse", command=self.browse_directory).grid(
            sticky="nsew", row=cur_row, column=2
        )
        cur_row += 1

        # set up session
        Button(self.window, text="Set Up Session", command=self.init_session).grid(
            sticky="nsew", row=cur_row, column=1
        )
        cur_row += 2

        ### control recording ###

        Label(self.window, text="Record: ").grid(sticky="w", row=cur_row, column=0)
        self.record_on = IntVar(value=-1)
        Radiobutton(
            self.window,
            text="Ready",
            selectcolor="blue",
            indicatoron=0,
            variable=self.record_on,
            value=0,
            state="disabled",
        ).grid(stick="nsew", row=cur_row, column=1)
        Radiobutton(
            self.window,
            text="On",
            selectcolor="green",
            indicatoron=0,
            variable=self.record_on,
            value=1,
            command=self.start_record,
        ).grid(sticky="nsew", row=cur_row + 1, column=1)
        Radiobutton(
            self.window,
            text="Off",
            selectcolor="red",
            indicatoron=0,
            variable=self.record_on,
            value=-1,
            command=self.stop_record,
        ).grid(sticky="nsew", row=cur_row + 2, column=1)
        Button(self.window, text="Save Video", command=lambda: self.save_vid()).grid(
            sticky="nsew", row=cur_row + 1, column=2
        )
        Button(
            self.window, text="Delete Video", command=lambda: self.save_vid(delete=True)
        ).grid(sticky="nsew", row=cur_row + 2, column=2)

        cur_row += 4

        ### close program ###

        Button(self.window, text="Close", command=self.closeGUI).grid(
            sticky="nsew", row=cur_row, column=0, columnspan=2
        )

        ### configure size of empty rows

        _, row_count = self.window.grid_size()
        for r in range(row_count):
            self.window.grid_rowconfigure(r, minsize=20)

    def run(self):

        self.window.mainloop()


def main():

    # import multiprocess as mp
    # mp.set_start_method("spawn")

    dlc_live_gui = DLCLiveGUI()
    dlc_live_gui.run()
