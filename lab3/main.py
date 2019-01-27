from mpi4py import MPI
import numpy
import time

com = MPI.COMM_WORLD
id = com.Get_rank()
size = com.Get_size()

def scatterMatrixByRows(comSize, matrix):
    rows = matrix.shape[0]
    counts = [rows // comSize for i in range(comSize)]
    if (rows % comSize != 0): counts[comSize - 1] += rows % comSize
    offset = 0
    result = []
    for c in counts:
        result.append(matrix[offset : offset + c])
        offset += c
    return result

n = 1000
matrix = None
vector = None
matrixParts = None
if id == 0:
    #numpy.random.seed(1)
    matrix = numpy.matrix(numpy.random.rand(n, n))
    vector = numpy.matrix(numpy.random.rand(1, n))
    # matrix = numpy.matrix([[2,4,0], [-2, 1, 3], [-1, 0, 1]])
    # vector = numpy.matrix([[1,2,-1]])
    matrixParts = scatterMatrixByRows(size, matrix) 
    time.perf_counter()
vector = com.bcast(vector)
nodeMatrixPart = com.scatter(matrixParts, 0)
nodeResult = []
for row in nodeMatrixPart:
    maultiply = vector @ row.transpose()
    nodeResult.append(maultiply.item(0))

result = com.gather(nodeResult)
if id == 0: 
    print(numpy.concatenate(result))
    print("Calculation takes ", time.perf_counter())