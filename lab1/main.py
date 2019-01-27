from mpi4py import MPI

class ProcessWork:
    actions = []
    data = []

    def doWork(self):
        return sum(self.data)

def buildProcessInfo(size, array):
    processStatus = [False for _ in range(size)]
    result = [ProcessWork() for _ in range(size)]
    activeProcesses = size
    datalength = len(array)
    #build work plan
    while activeProcesses > 1:
        for i in range(size):
            if processStatus[i]: continue

            processToComunicate = activeProcesses - i - 1
            if (processToComunicate == i):
                result[0].actions.append((False, i))
                result[i].actions.append((True, 0))
                activeProcesses //= 2
                processStatus[i] = True
                continue
            if (processToComunicate > 0):
                result[i].actions.append((True, 0))
                processStatus[i] = True
            if (i < activeProcesses // 2):
                result[i].actions.append((False, processToComunicate))
            elif i < activeProcesses:
                result[i].actions.append((True, processToComunicate))
                processStatus[i] = True
        activeProcesses //= 2
    #prepare data splitting
    counts = [datalength // size for i in range(size)]
    if (datalength % size != 0): counts[size - 1] += datalength % size
    offset = 0
    for i, c in enumerate(counts):
        result[i].data = array[offset : offset + c]
        offset += c
    return result

comm = MPI.COMM_WORLD
id = comm.Get_rank()
size = comm.Get_size()

comm.barrier()
print(f'Size is {size}')
print(f'Starting work {id}')

splitting = None
if id == 0:
    print('Prepare data')
    data = [i for i in range(1, 5)]
    splitting = buildProcessInfo(size, data)


processWork = comm.scatter(splitting, 0)
print(f'Process \'{id}\' work is {processWork.actions} ')
thisProcessSum = processWork.doWork()

for shouldSend, communicateWith in processWork.actions:
    if shouldSend:
        print(f'{id}: send')
        comm.send(thisProcessSum, dest=communicateWith)
    else:
        print(f'{id}: recieve')
        threadSum = comm.recv(souce=communicateWith)
        thisProcessSum += threadSum

if id == 0:
    print(f'Sum of sequence is {thisProcessSum}')

print(f'Work {id} finished')