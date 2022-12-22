import sys
sys.path.append('../02-vector/')
from vector import *

def angle(a,b):
	return math.acos(dot(a,b)/(length(a)*length(b)))

def area(a,b,c): # a,b,c 圍成的三角形面積 = cross(b-a, c-a)/2
	return cross(sub(b,a), sub(c,a))/2.0

def parallel(a,b): # 兩向量 a,b 是否平行
	return cross(a,b)==0

def distancePointLine(point, line): # 點到線的距離
	p, p1, p2 = point, line.p1, line.p2
	return abs(Cross(p-p1, p2-p1))/Distance(p1, p2)

def distancePointSegment(point, segment): # 點到線段的距離
	p, p1, p2 = point, line.p1, line.p2
	if dot(sub(p,p1),sub(p2,p1))<0 or dot(sub(p, p2, p1, p2))<0:
		return min(distance(p, p1), distance(p, p2))
	return distancePointLine(p, segment)

def projectionPointLine(point, line): # 點投射到線上的哪裡？
	p, p1, p2 = point, line.p1, line.p2
	k = dot(sub(p2,p1), sub(p, p1))/length(sub(p2, p1))
	return add(p1, sub(p2, p1))*k

def symmetryPointLine(point, line): # 點相對線的對稱點
	p, q = point, projectionPointLine(p, line) # q 是投射點
	return [2*q[0]-p[0], 2*q[1]-p[1]] # 對稱點在投射點延伸兩倍的位置


if __name__ == '__main__':
	print('cross([1,2], [2,1])=', cross([1.0,2.0], [2.0,1.0]))
	p0 = [0.0,0.0]; p1 = [1.0,1.0]; p2 = [2.0, 1.0]; p3 = [-1.0, 0.0]
	print('direction(p0, p1, p2)=', direction(p0, p1, p2))
	print('intersect(p0, p1, p2, p3)=', intersect(p0, p1, p2, p3))
	l1 = Line(p0, p1)
	print('relationPointLine(p2, l1)=', relationPointLine(p2, l1))
