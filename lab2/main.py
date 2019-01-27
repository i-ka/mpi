from mpi4py import MPI
from decimal import Decimal, getcontext
from math import factorial
import time

com = MPI.COMM_WORLD
id = com.Get_rank()
size = com.Get_size()

def chudnovsky(f, to):
    pipart = Decimal(0)
    for k in range(f, to + 1):
        pipart += (Decimal(-1)**k)*(Decimal(factorial(6*k))/((factorial(k)**3)*(factorial(3*k)))* (13591409+545140134*k)/(640320**(3*k)))
    return pipart

n = 1000
getcontext().prec = n

if id == 0:
    time.perf_counter()
    print("Calculate on", size," nodes")
pipart = chudnovsky(int((n / size * id) + 1), int(n / size * (id + 1)))
piparts = com.gather(pipart)

if id == 0:
    pi = Decimal(13591409)
    for part in piparts:
        pi += part
    pi = (pi * Decimal(10005).sqrt()/4270934400) ** -1
    print(pi)
    print("Calculation takes ", time.perf_counter())

