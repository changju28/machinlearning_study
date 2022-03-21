# numpy 행렬 만들기
import numpy as np
"""
import numpy as np

A = np.array([
    [1, -1, 2],
    [3, 2, 2],
    [4, 1, 2],
    [7, 5, 6]
])
print("{}행 {}열의 행렬 A".format(len(A), len(A[0])))
print(A)

B = np.array([
    [0, 1],
    [-1, 3],
    [5, 2]
])
print()
print("{}행 {}열의 행렬 B".format(len(B), len(B[0])))
print(B)

C = np.random.rand(3, 5)
print("{}행 {}열의 행렬 A".format(len(C), len(C[0])))
print(C)
print()
D = np.zeros((2, 4))
print("{}행 {}열의 행렬 A".format(len(D), len(D[0])))
print(D)
"""

# #numpy 행렬 더하기
"""
import numpy as np

# 행렬의 덧셈은 행과 열의 크기가 같아야 한다.
A = np.array([
    [1, 1],
    [2, 3]
])

B = np.array([
    [3, 1],
    [2, 4]
])

print(A + B)
"""
# 행렬곱셈

# _1 스칼라곱: 행열의 곱셈 각 원소에 숫자를 곱한 값

"""
i = 5

A = np.array([
    [3, 1],
    [2, 3]
])

print(A * i)
"""

# _2 내적곱 (A의 열과 B의 행의 수가 같아야 곱셈 가능)
"""

A = np.array([
    [1, 3, 1],
    [2, 2, 1]
])

B = np.array([
    [5, 6],
    [4, 2],
    [3, 1]
])

print(np.dot(A, B))
print(A @ B)
# AB는 A의 1행과 B의 1열의 원소들을 곱한 후 모두 더한 값을 원소로 나타내는 A의 행과 B의 열로 이루어진 행렬이 된다. EX) 2행 2열의 행렬
"""

# _3 요소별 곱(행과 열이 같은 두 행열의 원소들의 곱)
# AoB로 표기
"""
A = np.array([
    [1, 2],
    [3, 4],
])

B = np.array([
    [-1, 2],
    [3, 1],
])
print("AoB")
print("{}".format(A*B))
"""


# _4 전치 행렬(transposed matrix) : 행과 열을 바꾼것
# 목적: 행렬과 행렬을 곱할 때 왼쪽 행렬의 열 수랑 오른쪽 행렬의 행 수가 맞혀 행렬의 계산을 도와줌
"""
A = np.array([
    [1, 2, 1],
    [3, 2, 2]
])
At = np.transpose(A)
At1 = A.T

print("A\n{}\n 의 전치 행렬은 \n{}".format(A, At))
print(At1)
"""

# _5 단위 행렬(identity matrix) : 1행1열을 기준으로 대각석이 전부 1이고 나머지가 0인 정사각형의 행렬
# 목적: 행렬간에 곱할때 기존 행렬과 같은 값을 갖기 위한 행렬
"""
A = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

I = np.identity(3)

print(I)
"""

# _6 역행렬(inverse matrix) : 행렬간의 곱에서 결과값이 단위행렬이 나오게하는 행렬을 역행렬이라 함
# 모든 행렬의 역행렬이 있는것은 아니다
"""
A = np.array([
    [3, 4],
    [1, 2],
])

A_ = np.array([
    [1, -2],
    [-0.5, 1.5],
])

iver_A = np.linalg.pinv(A)
print(iver_A)

print(A @ A_)
"""