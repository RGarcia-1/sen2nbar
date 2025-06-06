import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="sen2nbar",
    version="2024.6.0",
    url="https://github.com/ESDS-Leipzig/sen2nbar",
    license="MIT",
    author="David Montero Loaiza",
    author_email="dml.mont@gmail.com",
    description="Nadir BRDF Adjusted Reflectance (NBAR) for Sentinel-2 in Python",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    package_data={"sen2nbar": ["data/*.json"]},
    install_requires=[
        "cubo>=2024.6.0",
        "pystac",
        "rasterio>=1.4.3",
        "requests",
        "rioxarray>=0.13.4",
        "scipy>=1.10.1",
        "tqdm",
        "defusedxml",
        "xmltodict",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)
