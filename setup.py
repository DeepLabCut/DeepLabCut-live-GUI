"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplabcut-live-gui",
    version="1.0",
    author="A. & M. Mathis Labs",
    author_email="adim@deeplabcut.org",
    description="GUI to run real time deeplabcut experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut-live-GUI",
    python_requires=">=3.5, <3.8",
    install_requires=[
        "deeplabcut-live",
        "pyserial",
        "pandas",
        "tables",
        "multiprocess",
        "imutils",
        "pillow",
        "tqdm",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points={
        "console_scripts": [
            "dlclivegui=dlclivegui.dlclivegui:main",
            "dlclivegui-video=dlclivegui.video:main",
        ]
    },
)
