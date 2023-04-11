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

data  = np.zeros(100, dtype = float)
value = np.zeros(100, dtype = float)

req_list = []

if rank == 0 :
    for i in range(100) :
        data[i] = i * 100
        req_send = comm.Isend(data[i:i+1], dest = 1, tag = i)
        req_list.append(req_send)
elif rank == 1 :
    for i in range(100) :
        req_recv = comm.Irecv(value[i:i+1] , source = 0, tag = i)
        req_list.append(req_recv)

MPI.Request.Waitall(req_list)

# if rank == 0 :
#     for i in range(100) :
#         data[i] = i * 100
#         req_send = comm.isend(data[i], dest = 1, tag = i)
#         req_list.append(req_send)
# elif rank == 1 :
#     for i in range(100) :
#         req_recv = comm.irecv( source = 0, tag = i)
#         req_list.append(req_recv)

# value = MPI.Request.waitall(req_list)

if rank == 0 :
    print("data[99] = {0}\n".format(data[99]))
if rank == 1 :
    print("value[99] = {0}\n".format(value[99]))
