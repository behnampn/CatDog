MIOpen implementation of 2 parallel convolutions

to run : make m1=<mask 1> m2=<mask 2> a1=<algo 1> a2=<algo 2>
eg: make m1=30 m2=30 a1=1 a2=1

Algo list:
0 - Direct
1 - Winograd
2 - GEMM

set environment variable MIOPEN_FIND_MODE before running code: export MIOPEN_FIND_MODE=1
