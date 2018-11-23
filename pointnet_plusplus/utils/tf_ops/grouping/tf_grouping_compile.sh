#!/usr/bin/env bash
source ../config.sh

$nvcc_bin tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
  -lcudart \
  -I $cuda_include_dir \
  -I $tensorflow_include_dir \
  -I $tensorflow_external_dir \
  -L $cuda_library_dir \
  -L $tensorflow_library_dir \
  -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

