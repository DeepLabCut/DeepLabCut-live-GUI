# DeepLabCut-Live! GUI <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1596193544929-NHMVMXPVEYZ6R4I45DSR/ke17ZwdGBToddI8pDm48kOHwsIwndRGCkvek0XAcW4ZZw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVH0wqgmu6zkAOZ3crWCtkmLcPIuzHaxU8QRzZwtrVtHupu3E9Ef3XsXP1C_826c-iU/DLCLIVEGUI_LOGO.png?format=750w" width="350" title="DLC-live GUI" alt="DLC LIVE! GUI" align="right" vspace = "100">

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
![PyPI - Python Version](https://img.shields.io/pypi/v/deeplabcut-live-gui)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deeplabcut-live-gui?color=purple)
![Python package](https://github.com/DeepLabCut/DeepLabCut-live/workflows/Python%20package/badge.svg)

[![License](https://img.shields.io/pypi/l/deeplabcutcore.svg)](https://github.com/DeepLabCut/deeplabcutlive/raw/master/LICENSE)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/deeplabcut)
[![Gitter](https://badges.gitter.im/DeepLabCut/community.svg)](https://gitter.im/DeepLabCut/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Twitter Follow](https://img.shields.io/twitter/follow/DeepLabCut.svg?label=DeepLabCut&style=social)](https://twitter.com/DeepLabCut)

GUI to run [DeepLabCut-live](https://github.com/DeepLabCut/DeepLabCut-live) on a video feed, record videos, and record external timestamps.

## [Installation Instructions](docs/install.md)

## Getting Started

#### Open DeepLabCut-live-GUI

In a terminal, activate the conda or virtual environment where DeepLabCut-live-GUI is installed, then run:

```
dlclivegui
```


#### Configurations


First, create a configuration file: select the drop down menu labeled `Config`, and click `Create New Config`. All settings, such as details about cameras, DLC networks, and DLC-live Processors, will be saved into configuration files so that you can close and reopen the GUI without losing all of these details. You can create multiple configuration files on the same system, so that different users can save different camera options, etc on the same computer. To load previous settings from a configuration file, please just select the file from the drop-down menu. Configuration files are stored at `$HOME/Documents/DeepLabCut-live-GUI/config`. These files do not need to be edited manually, they can be entirely created and edited automatically within the GUI.

#### Set Up Cameras <img src= https://imagizer.imageshack.com/img924/626/acJhWD.png align="right">

To setup a new camera, select `Add Camera` from the dropdown menu, and then click `Init Cam`. This will be bring up a new window where you need to select the type of camera (see [Camera Support](docs/camera_support.md)), input a name for the camera, and click `Add Camera`. This will initialize a new `Camera` entry in the drop down menu. Now, select your camera from the dropdown menu and click`Edit Camera Settings` to setup your camera settings (i.e. set the serial number, exposure, cropping parameters, etc; the exact settings depend on the specific type of camera). Once you have set the camera settings, click `Init Cam` to start streaming. To stop streaming data, click `Close Camera`, and to remove a camera from the dropdown menu, click `Remove Camera`.

#### Processor (optional)

To write custom `Processors`, please see [here](https://github.com/DeepLabCut/DeepLabCut-live/tree/master/dlclive/processor). The directory that contains your custom `Processor` should be a python module -- this directory must contain an `__init__.py` file that imports your custom `Processor`. For examples of how to structure a custom `Processor` directory, please see [here](https://github.com/DeepLabCut/DeepLabCut-live/tree/master/example_processors).

To use your processor in the GUI, you must first add your custom `Processor` directory to the dropdown menu: next to the `Processor Dir` label, click `Browse`, and select your custom `Processor` directory. Next, select the desired directory from the `Processor Dir` dropdown menu, then select the `Processor` you would like to use from the `Processor` menu. If you would like to edit the arguments for your processor, please select `Edit Proc Settings`, and finally, to use the processor, click `Set Proc`. If you have previously set a `Processor` and would like to clear it, click `Clear Proc`.

#### Configure DeepLabCut Network

<img src= https://imagizer.imageshack.com/img923/9730/MNzr1J.png align="right">

Select the `DeepLabCut` dropdown menu, and click `Add DLC`. This will bring up a new window to choose a name for the DeepLabCut configuration, choose the path to the exported DeepLabCut model, and set DeepLabCut-live settings, such as the cropping or resize parameters. Once configured, click `Update` to add this DeepLabCut configuration to the dropdown menu. You can edit the settings at any time by clicking `Edit DLC Settings`. Once configured, you can load the network and start performing inference by clicking `Start DLC`. If you would like to view the DeepLabCut pose estimation in real-time, select `Display DLC Keypoints`. You can edit the keypoint display settings (the color scheme, size of points, and the likelihood threshold for display) by selecting `Edit DLC Display Settings`.

If you want to stop performing inference at any time, just click `Stop DLC`, and if you want to remove a DeepLabCut configuration from the dropdown menu, click `Remove DLC`.

#### Set Up Session

Sessions are defined by the subject name, the date, and an attempt number. Within the GUI, select a `Subject` from the dropdown menu, or to add a new subject, type the new subject name in to the entry box and click `Add Subject`. Next, select an `Attempt` from the dropdown menu. Then, select the directory that you would like to save data to from the `Directory` dropdown menu. To add a new directory to the dropdown menu, click `Browse`. Finally, click `Set Up Session` to initiate a new recording. This will prepare the GUI to save data. Once you click `Set Up Session`, the `Ready` button should turn blue, indicating a session is ready to record.

#### Controlling Recording

If the `Ready` button is selected, you can now start a recording by clicking `On`. The `On` button will then turn green indicating a recording is active. To stop a recording, click `Off`. This will cause the `Ready` button to be selected again, as the GUI is prepared to restart the recording and to save the data to the same file. If you're session is complete, click `Save Video` to save all files: the video recording (as .avi file), a numpy file with timestamps for each recorded frame, the DeepLabCut poses as a pandas data frame (hdf5 file) that includes the time of each frame used for pose estimation and the time that each pose was obtained, and if applicable, files saved by the `Processor` in use. These files will be saved into a new directory at `{YOUR_SAVE_DIRECTORY}/{CAMERA NAME}_{SUBJECT}_{DATE}_{ATTEMPT}`

- YOUR_SAVE_DIRECTORY : the directory chosen from the `Directory` dropdown menu.
- CAMERA NAME : the name of selected camera (from the `Camera` dropdown menu).
- SUBJECT : the subject chosen from the `Subject` drowdown menu.
- DATE : the current date of the experiment.
- ATTEMPT : the attempt number chosen from the `Attempt` dropdown.

If you would not like to save the data from the session, please click `Delete Video`, and all data will be discarded. After you click `Save Video` or `Delete Video`, the `Off` button will be selected, indicating you can now set up a new session.

#### References:

If you use this code we kindly ask you to you please [cite Kane et al, eLife 2020](https://elifesciences.org/articles/61909). The preprint is available here: https://www.biorxiv.org/content/10.1101/2020.08.04.236422v2
