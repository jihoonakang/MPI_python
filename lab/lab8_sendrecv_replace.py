################################################################################
# MPI python example
#
# Copyright 2023. Ji-Hoon Kang, All rights reserved.
# This project is released under the terms of the MIT License (see LICENSE )
################################################################################

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

a = np.array([rank + 1], dtype = int)

inext = rank + 1
iprev = rank - 1

if rank == 0 :
    iprev = size - 1
if rank == size - 1 :
    inext = 0

for i in range(size) :
    if rank == i :
        print('BEFORE : myrank={0}, a = {1}'.format(rank, a))

stime = MPI.Wtime()

comm.Sendrecv_replace(a, inext, 77, iprev, 77)

print(rank, MPI.Wtime() - stime)

for i in range(size) :
    if rank == i :
        print('AFTER  : myrank={0}, a = {1}'.format(rank, a))
