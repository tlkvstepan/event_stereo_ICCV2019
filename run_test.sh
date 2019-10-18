#!/bin/bash

# Runs tests in a non-Brazil setup.

# To be safe, switch into the folder that contains this script.
cd "$( cd "$( dirname "$0" )" && pwd )"

PROJECT_PATHS=./src
PROJECT_PATHS=./third_party/PracticalDeepStereo_NIPS2018:$PROJECTPATHS

env PYTHONPATH=$PROJECT_PATHS python -m pytest test -v
