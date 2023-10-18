################################################################################
# MPI python example
#
# Copyright 2023. Ji-Hoon Kang, All rights reserved.
# This project is released under the terms of the MIT License (see LICENSE )
################################################################################

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank() # myrank = comm.rank

buf_size = 100000
a = np.ones(buf_size, dtype = int)
b = np.empty(buf_size, dtype = int)

if rank == 0 :
    comm.Send(a, dest = 1, tag = 11)
    comm.Recv(b, source = 1, tag = 55)

elif rank ==1 :
    comm.Send(a, dest = 0, tag = 55)
    comm.Recv(b, source = 0, tag = 11)

print("Everything okay")