"""
이번 과제에서는 지금까지 배운 모든 내용들을 종합해서 복잡한 행렬 연산을 numpy를 사용해서 계산해보겠습니다.
행렬 A, B, C, D는  저희가 이미 정의해놨고요. 여러분들은 지금까지 배운 내용을 종합하셔서 아래 행렬 연산을 numpy로 작성하시면 됩니다.

Bt * (2 * At) * (3 * C(-1) + Dt

"""
import numpy as np

A = np.array([
    [1, -1, 2],
    [3, 2, 2]
])

B = np.array([
    [0, 1],
    [-1, 1],
    [5, 2]
])

C = np.array([
    [2, -1],
    [-3, 3]
])

D = np.array([
    [-5, 1],
    [2, 0]
])

# 행렬 연산을 result 변수에 저장하세요
result = B.T @ (2*A.T) @ (3*np.linalg.pinv(C) + D.T)

print(result)