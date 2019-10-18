import os

import numpy as np

from setuptools import setup, Extension, find_packages

# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk('configuration'):
    data_files.append((os.path.relpath(root, 'configuration'),
                       [os.path.join(root, f) for f in files]))

# Declare your scripts:
scripts = []

setup(
    name="dense_deep_event_stereo",
    version="1.0",

    # declare your packages
    packages=find_packages(where="src", exclude=("test", )),
    package_dir={"": "src"},

    # include data files
    data_files=data_files,

    # declare your scripts
    scripts=scripts,

    # set up the shebang
    options={
        # make sure the right shebang is set for the scripts - use the
        # environment default Python
        'build_scripts': {
            'executable': '/apollo/sbin/envroot "$ENVROOT/bin/python"',
        },
    },
    setup_requires=[
        'setuptools>=18.0',
        'cython',
    ],
    ext_modules=[
        Extension(
            'dense_deep_event_stereo._events',
            ["src/dense_deep_event_stereo/_events.pyx"],
            include_dirs=[np.get_include()]),
    ],

)
