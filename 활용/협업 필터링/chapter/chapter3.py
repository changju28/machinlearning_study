import numpy as np

# 노트에 있는 코드를 바탕으로 다양한 실습을 해보세요

A = np.array([
    [3, 3, 2, 3, 1],
    [5, 2, 2, 3, 1],
    [3, 3, 2, 3, 1],
    [3, 1, 4, 3, 1],
])

# 행 리턴
print(A[3])
print(A[0])

# 열 리턴

print(A[:, 3])
print(A[:, 0])

print(A[1, 3])