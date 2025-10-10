import random
import time
import math

def function(x: float) -> float:
    return pow(x,2)   

def cost(mat: list[list[float]], perm: list) -> float:
    
def main():
    random.seed()
    N = 100
    mat = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            mat[i][j] = random.randint(10, 1000)
            mat[j][i] = random.randint(10,1000)

    perm = range(N)
    start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start


    


if __name__ == '__main__':
    main()