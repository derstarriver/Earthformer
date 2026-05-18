#!/usr/bin/env python
import io
import os
import re
from datetime import datetime
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('src', 'earthformer', '__init__.py')

if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')

requirements = [
    'boto3',
    'scipy',
    'tqdm',
    'requests',
    'tensorboard',
    'einops>=0.3.0',
    'omegaconf',
    'matplotlib',
    'packaging',
]

setup(
    name='earthformer',
    version=VERSION,
    python_requires='>=3.6',
    description='Earthformer: Exploring Space-Time Transformers for Earth System Forecasting - ENSO/SST Prediction',
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    packages=find_packages(where="src", exclude=('tests', 'scripts')),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
)
