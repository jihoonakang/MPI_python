# for x in range(4):
#     y = x*x + x + 1
#     print("The value of x*x+x+1 = {0}, x = {1}".format(y, x))

# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# myrank = comm.Get_rank() # myrank = comm.rank
# nprocs = comm.Get_size() # nprocs = comm.size

# if(myrank==0) :
#     x=0.0
# elif(myrank==1) :
#     x=1.0
# elif(myrank==2) :
#     x=2.0
# elif(myrank==3) :
#     x=3.0
# y=x*x+x+1
# print("process{0} of {1} : the value of x*x+x+1 = {2}, x = {3} ".
#       format(myrank,nprocs,y,x))

from mpi4py import MPI
comm = MPI.COMM_WORLD
myrank = comm.Get_rank() # myrank = comm.rank
nprocs = comm.Get_size() # nprocs = comm.size

x =myrank
y=x*x+x+1

print("process{0} of {1} : the value of x*x+x+1 = {2}, x = {3}".
      format(myrank,nprocs,y,x))
