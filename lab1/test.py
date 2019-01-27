from mpi4py import MPI

comm = MPI.COMM_WORLD
print(F'size is {comm.Get_size()}')
print(F'Hello from {comm.Get_rank()}')