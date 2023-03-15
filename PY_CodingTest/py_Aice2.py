# DataFrame : 2차원(col[행]과 row[열]) 테이블 데이터 구조를 가진 자료형
# 딕셔너리, 리스트, 파일을 통해 생성할 수 있다.
# read_csv 함수를 이용해서 생성 가능
# .head 위에서 부터 .tail 밑에서 부터 확인
# shape : row와 col의 개수를 튜플로 반환, columns : 컬럼명을 확인할 수 있음
# info : 데이터 타입, 각 아이템의 개수 등을 출력함
# describe 데이터 요약 통계량을 나타낸다, dtype : 데이터 형태의 종류
import pandas as pd

print("Dictionary 형태로 데이터 프레임 생성")
a1 = pd.DataFrame({"a":[1,2,3], "b": [4,5,6],"c":[7,8,9]})
print(a1) # Dictionary 형으로 생성

print("List 형태로 데이터 프레임 생성")
a2 = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], ["a","b","c"])
print(a2) # List 형태로 데이터 프레임 생성

# 파일 읽어서 데이터프레임 생성 : pandas.read_csv 함수 사용
# cust.head(n=3) 위에서 부터 3개의 데이터 가져온다.
# cust.tail(n=3) 위에서 부터 3개의 데이터 가져온다.







