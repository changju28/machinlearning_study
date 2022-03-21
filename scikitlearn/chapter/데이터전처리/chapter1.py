# 필요한 도구 임포트
from sklearn import preprocessing
import pandas as pd

PATIENT_FILE_PATH = './datasets/liver_patient_data.csv'
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# 데이터 파일을 pandas dataframe으로 가지고 온다
liver_patients_df = pd.read_csv(PATIENT_FILE_PATH)

# Normalization할 열 이름들
features_to_normalize = ['Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase']

# 코드를 쓰세요

scaler = preprocessing.MinMaxScaler()

normalization_data = scaler.fit_transform(liver_patients_df[features_to_normalize])

normalized_df = pd.DataFrame(normalization_data, columns=features_to_normalize)

# 체점용 코드
print(normalized_df.describe())