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
matrix1 = None
matrix2 = None
nodeMatrix2Rows = None
matrix2Parts = None
checkMatrix = None
if id == 0:
    #numpy.random.seed(1)
    matrix1 = numpy.matrix(numpy.random.rand(n, n), copy = False)
    matrix2 = numpy.matrix(numpy.random.rand(n, n), copy = False)
    checkMatrix = matrix1 @ matrix2
    matrix2 = matrix2.transpose()
    # matrix1 = numpy.matrix([[2, 4, 0], [-2, 1, 3], [-1, 0, 1]])
    # matrix2 = numpy.matrix([[2, 4, 0], [-2, 1, 3], [-1, 0, 1]]).transpose()
    matrix2Parts = scatterMatrixByRows(size, matrix2)
    time.perf_counter()

matrix1 = com.bcast(matrix1, 0)
nodeMatrix2Rows = com.scatter(matrix2Parts, 0)
nodeResult = []
for col in range(0, nodeMatrix2Rows.shape[0]):
    m2 = nodeMatrix2Rows[col].transpose()
    for row in range(0, matrix1.shape[0]):
        m1 = matrix1[row]
        mul = (m1 @ m2).item(0)
        nodeResult.append(mul)

result = com.gather(nodeResult)
if id == 0:
    result = numpy.matrix(numpy.concatenate(result))
    result.shape = (n, n)
    result = result.transpose()
    print(result)
    print("Calculation takes ", time.perf_counter())
    print("Checking...")
    right = numpy.allclose(result, checkMatrix, .000000001)
    print("Ok!" if right else "Not ok :(")