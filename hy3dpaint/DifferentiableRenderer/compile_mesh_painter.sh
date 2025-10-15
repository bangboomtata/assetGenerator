#!/bin/bash

PYTHON=$(which python)
CXX=c++

$CXX -O3 -Wall -shared -std=c++11 -fPIC \
  `$PYTHON -m pybind11 --includes` \
  mesh_inpaint_processor.cpp \
  -o mesh_inpaint_processor$( $PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" )