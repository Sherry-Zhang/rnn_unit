#!/bin/sh
#export KMP_BLOCKTIME=1
#export KMP_AFFINITY=verbose,granularity=core,noduplicates,compact,0,0
export KMP_AFFINITY=granularity=core,noduplicates,compact,0,0
export OMP_NUM_THREADS=56   #SKX
#export OMP_NUM_THREADS=40   #SKX-6148
#export OMP_NUM_THREADS=68   #KNL
#export OMP_NUM_THREADS=72   #KNM
#export OMP_NUM_THREADS=44   #BDW
./sru
