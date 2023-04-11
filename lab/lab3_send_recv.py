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

if rank == 0 :
    data = np.full(100, 3.0, dtype = float)
    comm.Send(data, dest = 1, tag = 55)

elif rank ==1 :
    value = np.empty(100, dtype = float)
    status = MPI.Status()
    comm.Recv(value, source = MPI.ANY_SOURCE, tag = 55, status = status)
    
    print("p{0} got data from processor {1}".format(rank, status.source))
    print("p{0} got {1} byte".format(rank, status.count))
    print("p{0} values(5) = {1}".format(rank, value[5]))
