"""Setup configuration for the DeepLabCut Live GUI."""

from __future__ import annotations

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplabcut-live-gui",
    version="2.0",
    author="A. & M. Mathis Labs",
    author_email="adim@deeplabcut.org",
    description="PyQt-based GUI to run real time DeepLabCut experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut-live-GUI",
    python_requires=">=3.10",
    install_requires=[
        "deeplabcut-live",
        "PyQt6",
        "numpy",
        "opencv-python",
        "vidgear[core]",
    ],
    extras_require={
        "basler": ["pypylon"],
        "gentl": ["harvesters"],
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points={
        "console_scripts": [
            "dlclivegui=dlclivegui.gui:main",
        ]
    },
)
