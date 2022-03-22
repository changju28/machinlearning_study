import numpy as np
from math import sqrt


def distance(user_1, user_2):
    """유클리드 거리를 계산해주는 함수"""
    # 코드를 쓰세요

    return sqrt(((user_1 - user_2) ** 2).sum())


# 실행 코드
user_1 = np.array([0, 1, 2, 3, 4, 5])
user_2 = np.array([0, 1, 4, 6, 1, 4])

print(distance(user_1, user_2))

