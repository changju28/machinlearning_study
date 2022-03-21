"""
이번 과제에서는 위 결정 트리를 if-else 문을 이용해서 구현해보겠습니다.

survival_classifier함수
survival_classifier 함수는 안전 벨트를 했는지를 나타내는 불린형 파라미터, seat_belt, 고속도로였는지를 나타내는 불린형 파라미터, highway, 사고 당시 주행속도를 나타내는 숫자형 파라미터, speed, 사고자 나이를 나타내는 숫자형 파라미터, age를 받습니다.

그리고 위에 나와 있는 결정 트리대로 교통 사고 데이터가 생존할 건지 사망할 건지를 리턴합니다. (생존을 예측할 시 0을 리턴하고, 사망을 예측할 시 1을 리턴합니다)

"""
def survival_classifier(seat_belt, highway, speed, age):
    if seat_belt:
        return 0
    else:
        if not highway:
            return 0
        else:
            if speed < 100:
                return 0
            else:
                if age < 50:
                    return 0
                else:
                    return 1

# 코드를 쓰세요


print(survival_classifier(False, True, 110, 55))
print(survival_classifier(True, False, 40, 70))
print(survival_classifier(False, True, 80, 25))
print(survival_classifier(False, True, 120, 60))
print(survival_classifier(True, False, 30, 20))