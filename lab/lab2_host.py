################################################################################
# MPI python example
#
# Copyright 2023. Ji-Hoon Kang, All rights reserved.
# This project is released under the terms of the MIT License (see LICENSE )
################################################################################

from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank() # myrank = comm.rank
nprocs = comm.Get_size() # nprocs = comm.size

ver, subver = MPI.Get_version()
if myrank == 0 :
    print("MPI Version {0}.{1}".format(ver, subver))

procName = MPI.Get_processor_name()

print("Hello World.(Process name={0}, nRank={1}, nProcs={2})".format(procName, myrank, nprocs))
