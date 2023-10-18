################################################################################
# MPI python example
#
# Copyright 2023. Ji-Hoon Kang, All rights reserved.
# This project is released under the terms of the MIT License (see LICENSE )
################################################################################

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

a = np.zeros(size, dtype = int)
b = np.zeros(size, dtype = int)

a[rank] = rank + 1

inext = rank + 1
iprev = rank - 1

if rank == 0 :
    iprev = size - 1
if rank == size - 1 :
    inext = 0

for i in range(size) :
    if rank == i :
        print('BEFORE : myrank={0}, A = {1}'.format(rank, a))

comm.Sendrecv(a, inext, 77, b, iprev, 77)
# b[rank] = comm.sendrecv(a[rank], inext, 77, None, iprev, 77)

for i in range(size) :
    if rank == i :
        print('AFTER  : myrank={0}, B = {1}'.format(rank, b))
