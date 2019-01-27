import time
import random

numbers = [random.randint(-1e4, 1e4) for _ in range(1000000)]

time.perf_counter()
print(sum(numbers))
print(f'Calculation takes {time.perf_counter()}')