## Camera Support

### Windows
- **The Imaging Source USB3 Cameras**: via code based on [Windows code samples](https://github.com/TheImagingSource/IC-Imaging-Control-Samples) provided by The Imaging Source. To use The Imaging Source USB3 cameras on Windows, you must first [install their drivers](https://www.theimagingsource.com/support/downloads-for-windows/device-drivers/icwdmuvccamtis/) and [C library](https://www.theimagingsource.com/support/downloads-for-windows/software-development-kits-sdks/tisgrabberdll/).
- **OpenCV compatible cameras**: OpenCV is installed with DeepLabCut-live-GUI, so webcams or other cameras compatible with OpenCV on Windows require no additional installation.

### Linux and NVIDIA Jetson Development Kits

- **OpenCV compatible cameras**: We provide support for many webcams and industrial cameras using OpenCV via Video4Linux drivers. This includes The Imaging Source USB3 cameras (and others, but untested). OpenCV is installed with DeepLabCut-live-GUI.
- **Aravis Project compatible USB3Vision and GigE Cameras**: [The Aravis Project](https://github.com/AravisProject/aravis) supports a number of popular industrial cameras used in neuroscience, including The Imaging Source, Point Grey, and Basler cameras. To use Aravis Project drivers, please follow their [installation instructions](https://github.com/AravisProject/aravis#installing-aravis). The Aravis Project drivers are supported on the NVIDIA Jetson platform, but there are known bugs (e.g. [here](https://github.com/AravisProject/aravis/issues/324)).

### Contributing New Camera Types

Any camera that can be accessed through python (e.g. if the company offers a python package) can be integrated into the DeepLabCut-live-GUI. To contribute, please build off of our [base `Camera` class](../dlclivegui/camera/camera.py), and please use our [currently supported cameras](../dlclivegui/camera) as examples.

New camera classes must inherit our base camera class, and provide at least two arguments:

- id: an arbitrary name for a camera
- resolution: the image size

Other common options include:

- exposure
- gain
- rotate
- crop
- fps

If the camera does not have it's own display module, you can use our Tkinter video display built into the DeepLabCut-live-GUI by passing `use_tk_display=True` to the base camera class, and control the size of the displayed image using the `display_resize` parameter (`display_resize=1` for full image, `display_resize=0.5` to display images at half the width and height of recorded images).

Here is an example of a camera that allows users to set the resolution, exposure, and crop, and uses the Tkinter display:

```python
from dlclivegui import Camera

class MyNewCamera(Camera)

    def __init__(self, id="", resolution=[640, 480], exposure=0, crop=None, display_resize=1):
        super().__init__(id,
                         resolution=resolution,
                         exposure=exposure,
                         crop=crop,
                         use_tk_display=True,
                         display_resize=display_resize)

```

All arguments of your camera's `__init__` method will be available to edit in the GUI's `Edit Camera Settings` window. To ensure that you pass arguments of the correct data type, it is helpful to provide default values for each argument of the correct data type (e.g. if `myarg` is a string, please use `myarg=""` instead of `myarg=None`). If a certain argument has only a few possible values, and you want to limit the options user's can input into the `Edit Camera Settings` window, please implement a `@static_method` called `arg_restrictions`. This method should return a dictionary where the keys are the arguments for which you want to provide value restrictions, and the values are the possible values that a specific argument can take on. Below is an example that restrictions the values for `use_tk_display` to `True` or `False`, and restricts the possible values of `resolution` to `[640, 480]` or `[320, 240]`.

```python
    @static_method
    def arg_restrictions():
        return {'use_tk_display' : [True, False],
                'resolution' : [[640, 480], [320, 240]]}
```

In addition to an `__init__` method that calls the `dlclivegui.Camera.__init__` method, you need to overwrite the `dlclivegui.Camera.set_capture_device`, `dlclive.Camera.close_capture_device`, and one of the following two methods: `dlclivegui.Camera.get_image` or `dlclivegui.Camera.get_image_on_time`.

Your camera class's `set_capture_device` method should open the camera feed and confirm that the appropriate settings (such as exposure, rotation, gain, etc.) have been properly set. The `close_capture_device` method should simply close the camera stream. For example, see the [OpenCV camera](../dlclivegui/camera/opencv.py) `set_capture_device` and `close_capture_device` method.

If you're camera has built in methods to ensure the correct frame rate (e.g. when grabbing images, it will block until the next image is ready), then overwrite the `get_image_on_time` method. If the camera does not block until the next image is ready, then please set the `get_image` method, and the base camera class's `get_image_on_time` method will ensure that images are only grabbed at the specified frame rate.

The `get_image` method has no input arguments, but must return an image as a numpy array. We also recommend converting images to 8-bit integers (data type `uint8`).

The `get_image_on_time` method has no input arguments, but must return an image as a numpy array (as in `get_image`) and the timestamp at which the image is returned (using python's `time.time()` function).

### Camera Specific Tips for Installation & Use:

#### Basler cameras

Basler USB3 cameras are compatible with Aravis. However, integration with DeepLabCut can also be obtained with `pypylon`, the python module to drive Basler cameras, and supported by the company. Please note using `pypylon` requires you to install Pylon viewer, a free of cost GUI also developed and supported by Basler and available on several platforms.

* **Pylon viewer**: https://www.baslerweb.com/en/sales-support/downloads/software-downloads/#type=pylonsoftware;language=all;version=all
* `pypylon`: https://github.com/basler/pypylon/releases

If you want to use DeepLabCut with a Basler USB3 camera via pypylon, see the folllowing instructions. Please note this is tested on Ubuntu 20.04. It may (or may not) work similarly in other platforms (contributed by [@antortjim](https://github.com/antortjim)). This procedure should take around 10 minutes:

**Install Pylon viewer**

1. Download .deb file
Download the .deb file in the downloads center of Basler. Last version as of writing this was **pylon 6.2.0 Camera Software Suite Linux x86 (64 Bit) - Debian Installer Package**.


2. Install .deb file

```
sudo dpkg -i pylon_6.2.0.21487-deb0_amd64.deb
```

**Install swig**

Required for compilation of non python code within pypylon

1. Install swig dependencies

You may have to install these in a fresh Ubuntu 20.04 install

```
sudo apt install gcc g++
sudo apt install libpcre3-dev
sudo apt install make
```

2. Download swig

Go to http://prdownloads.sourceforge.net/swig/swig-4.0.2.tar.gz and download the tar gz

3. Install swig
```
tar -zxvf swig-4.0.2.tar.gz
cd swig-4.0.2
./configure
make
sudo make install
```

**Install pypylon**

1. Download pypylon

```
wget https://github.com/basler/pypylon/archive/refs/tags/1.7.2.tar.gz
```

or go to https://github.com/basler/pypylon/releases and get the version you want!

2. Install pypylon

```
tar -zxvf 1.7.2.tar.gz
cd pypylon-1.7.2
python setup.py install
```

Once you have completed these steps, you should be able to call your Basler camera from DeepLabCut using the BaslerCam camera type that appears after clicking "Add camera")
