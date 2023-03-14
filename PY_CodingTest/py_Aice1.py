# Indexing : 무엇인가 가르킨다를 의미
# 인덱싱-> 특정한 값을 뽑아내는 역할
x = "Rome is not built in a day!"
print(x[0])
print(x[-1])
print(x[-6])
print(x[12])
# 슬라이싱(Slicing) : 원하는 단어를 뽑아내고자 할 때 사용
# x[시작 번호 : 끝번호]
print("슬라이싱 연습")
print(x[0:4])
print(x[12:])
print(x[:7])

# python 대표 자료형 String(문자열), integer(정수형), float(소수점 자리)
# List 자료형 : 순서대로 정리된 항목들을 담는 구조(특정 데이터를 반복적으로 처리하는데 특화)
a = [1,2,3] # 리스트 자료형
b = list(range(1,10,2))

# Tuple 자료형 :  ()으로 둘러쌈 a=(1,2,3), b = (2,)
# Pandas 자료형
print(a)

c = ['red', 'green', 'blue']
c.append('yellow') # 리스트의 끝에 yellow라는 요소 추가
print(c)

c.insert(1, 'black') # 'black'이라는 요소를 1번째 자리에 넣는다.
print(c)

d = ['purple', 'white']
c.extend(d) # d의 요소들을 c에 넣는다.
print(c)

q = [10, 20, 30, 40,50,60,70,80,90,100]
q.remove(90) # 90이라는 요소 삭제
print(q)

list1 = ['a','bb','c','d','aaa','c','ddd','aaa','b','cc','d','aaa',]
print(list1.count('aaa')) # aaa라는 것의 개수를 알 수 있다.

list2 = [1,-7,5,8,3,9,11,13]
list2.sort()
print(list2)

list2.sort(reverse=True)
print(list2)



