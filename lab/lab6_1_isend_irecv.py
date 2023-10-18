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

data  = []
value = []

req_list = []

if rank == 0 :
    for i in range(100) :
        data.append(i * 100)
        req_send = comm.isend(data[i], dest = 1, tag = i)
        req_list.append(req_send)
elif rank == 1 :
    for i in range(100) :
        req_recv = comm.irecv( source = 0, tag = i)
        req_list.append(req_recv)

value = MPI.Request.waitall(req_list)

if rank == 0 :
    print("data[99] = {0}\n".format(data[99]))
if rank == 1 :
    print("value[99] = {0}\n".format(value[99]))
