import numpy as np

# 체점을 위해 numpy에서 임의성 도구들의 결과가 일정하게 나오도록 해준다
np.random.seed(5)


def initialize(R, num_features):
    """임의로 유저 취향과 상품 속성 행렬들을 만들어주는 함수"""
    num_users, num_items = R.shape  # 유저 데이터 개수와 영화 개수를 변수에 저장

    # 코드를 쓰세요.
    print(num_items)
    print(num_users)
    Theta = np.random.rand(num_users, num_features)
    X = np.random.rand(num_features, num_items)
    return Theta, X


# 실제 값 행렬
R = np.array([
    [3, 4, 1, np.nan, 1, 2],
    [4, 4, 3, np.nan, 5, 3],
    [2, 3, np.nan, 1, 3, 4],
    [1, 3, 2, 4, 2, 2],
    [1, 2, np.nan, 2, 2, 4],
])

Theta, X = initialize(R, 2)
print(Theta, X)

