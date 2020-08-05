"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import tkinter as tk
from tkinter import ttk
from distutils.util import strtobool


class SettingsWindow(tk.Toplevel):
    def __init__(
        self,
        title="Edit Settings",
        settings={},
        names=None,
        vals=None,
        dtypes=None,
        restrictions=None,
        parent=None,
    ):
        """ Create a tkinter settings window
        
        Parameters
        ----------
        title : str, optional
            title for window
        settings : dict, optional
            dictionary of settings with keys = setting names.
            The value for each setting should be a dictionary with three keys:
            value (a default value),
            dtype (the data type for the setting),
            restriction (a list of possible values the parameter can take on) 
        names : list, optional
            list of setting names, by default None
        vals : list, optional
            list of default values, by default None
        dtypes : list, optional
            list of setting data types, by default None
        restrictions : dict, optional
            dictionary of setting value restrictions, with keys = setting name and value = list of restrictions, by default {}
        parent : :class:`tkinter.Tk`, optional
            parent window, by default None
        
        Raises
        ------
        ValueError
            throws error if neither settings dictionary nor setting names are provided
        """

        super().__init__(parent)
        self.title(title)

        if settings:
            self.settings = settings
        elif not names:
            raise ValueError(
                "No argument names or settings dictionary. One must be provided to create a SettingsWindow."
            )
        else:
            self.settings = self.create_settings_dict(names, vals, dtypes, restrictions)

        self.cur_row = 0
        self.combobox_width = 15

        self.create_window()

    def create_settings_dict(self, names, vals=None, dtypes=None, restrictions=None):
        """Create dictionary of settings from names, vals, dtypes, and restrictions

        Parameters
        ----------
        names : list
            list of setting names
        vals : list
            list of default setting values
        dtypes : list
            list of setting dtype
        restrictions : dict
            dictionary of settting restrictions
        
        Returns
        -------
        dict
            settings dictionary with keys = names and value = dictionary with value, dtype, restrictions
        """

        set_dict = {}
        for i in range(len(names)):

            dt = dtypes[i] if dtypes is not None else None

            if vals is not None:
                val = dt(val) if type(dt) is type else [dt[0](v) for v in val]
            else:
                val = None

            restrict = restrictions[names[i]] if restrictions is not None else None

            set_dict[names[i]] = {"value": val, "dtype": dt, "restriction": restrict}

        return set_dict

    def create_window(self):
        """ Create settings GUI widgets
        """

        self.entry_vars = []
        names = tuple(self.settings.keys())
        for i in range(len(names)):

            this_setting = self.settings[names[i]]

            tk.Label(self, text=names[i] + ": ").grid(row=self.cur_row, column=0)

            v = this_setting["value"]
            if type(this_setting["dtype"]) is list:
                v = [str(x) if x is not None else "" for x in v]
                v = ", ".join(v)
            else:
                v = str(v) if v is not None else ""
            self.entry_vars.append(tk.StringVar(self, value=v))

            use_restriction = False
            if "restriction" in this_setting:
                if this_setting["restriction"] is not None:
                    use_restriction = True

            if use_restriction:
                ttk.Combobox(
                    self,
                    textvariable=self.entry_vars[-1],
                    values=this_setting["restriction"],
                    state="readonly",
                    width=self.combobox_width,
                ).grid(sticky="nsew", row=self.cur_row, column=1)
            else:
                tk.Entry(self, textvariable=self.entry_vars[-1]).grid(
                    sticky="nsew", row=self.cur_row, column=1
                )

            self.cur_row += 1

        self.cur_row += 1
        tk.Button(self, text="Update", command=self.update_vals).grid(
            sticky="nsew", row=self.cur_row, column=1
        )
        self.cur_row += 1
        tk.Button(self, text="Cancel", command=self.destroy).grid(
            sticky="nsew", row=self.cur_row, column=1
        )

        _, row_count = self.grid_size()
        for r in range(row_count):
            self.grid_rowconfigure(r, minsize=20)

    def update_vals(self):

        names = tuple(self.settings.keys())

        for i in range(len(self.entry_vars)):

            name = names[i]
            val = self.entry_vars[i].get()
            dt = (
                self.settings[name]["dtype"] if "dtype" in self.settings[name] else None
            )

            val = [v.strip() for v in val.split(",")]
            use_dt = dt if type(dt) is type else dt[0]
            use_dt = strtobool if use_dt is bool else use_dt

            try:
                val = [use_dt(v) if v else None for v in val]
            except TypeError:
                pass

            val = val if type(dt) is list else val[0]

            self.settings[name]["value"] = val

        self.quit()
        self.destroy()

    def get_values(self):

        val_dict = {}
        names = tuple(self.settings.keys())
        for i in range(len(self.settings)):
            val_dict[names[i]] = self.settings[names[i]]["value"]

        return val_dict
