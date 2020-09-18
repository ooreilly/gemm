# GEMM
This repository contains my attempts at writing a fast single precision matrix-matrix multiplication kernel. The
kernel found in the file `optimized.cuh` experiments with techniques such as 128 bit wide loads, and
double buffering. Unfortunately, it is not extensively tested and in fact produces incorrect results
for certain problem sizes. For example,
```
./gemm 1000 1000 1000
gemm_baseline took:  1.63526 ms 
gemm_16x16 took:  0.221728 ms 
error at i, j = 0, 832, diff=2.14748e+09, A=2.59741e+10, B=0,  
Maximum difference: 2.14748e+09 

``
