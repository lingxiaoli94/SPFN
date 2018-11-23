#!/usr/bin/env bash
source ../config.sh

# TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC \
  -lcudart \
  -I $cuda_include_dir \
  -I $tensorflow_include_dir \
  -I $tensorflow_external_dir \
  -L $cuda_library_dir \
  -L $tensorflow_library_dir \
  -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

