#!/bin/sh
#lscpu
#rm sru
#icc test_sru.c sru.c  -fno-alias -restrict -qopt-prefetch=3 -xCOMMON-AVX512 -qopt-report=5 -qopenmp -lpthread -lm -lmkl_rt -ldl -O3 -g -o sru
icpc test_sru.c sru.c  -fno-alias -restrict -qopt-prefetch=3 -qopt-report=5 -qopenmp -lpthread -lm -lmkl_rt -ldl -O3 -march=native -g -o sru
#export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=core,noduplicates,compact,0,0
#bdw
#export OMP_NUM_THREADS=44
#skx6148
export OMP_NUM_THREADS=40
#knl
#export OMP_NUM_THREADS=68
#skx8180
#export OMP_NUM_THREADS=56
#knm
#export OMP_NUM_THREADS=72

./sru  

