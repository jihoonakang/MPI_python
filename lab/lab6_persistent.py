################################################################################
# MPI python example
#
# Copyright 2023. Ji-Hoon Kang, All rights reserved.
# This project is released under the terms of the MIT License (see LICENSE )
################################################################################

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

a = np.zeros(5, dtype = int)
b = np.empty(5, dtype = int)

rank = comm.Get_rank() # myrank = comm.rank

prequest = []
prequest.append(comm.Send_init(a, 1, 11))
prequest.append(comm.Recv_init(b, 0, 11))

for i in range(10) :
    if(rank == 0) :
        for j in range(5) :
            a[j] = a[j] + j + 1

    prequest[rank].Start()

    MPI.Request.Wait(prequest[rank])

    if(rank == 1) :
        print(b)
