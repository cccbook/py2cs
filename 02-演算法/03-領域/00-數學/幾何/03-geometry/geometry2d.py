from vector import *
from math import *

class Line: # 線
	def __init__(self, p1, p2):
		self.p1 = p1
		self.p2 = p2

def cross(a,b): # (外積) 叉積 a0*b1-a1*b0 (只適用於二維向量)
	return a[0]*b[1]-a[1]*b[0]

def normal(a): # 向量 a 的法向量
	return [-a[1]/length(a), a[0]/length(a)]

def angleLine(line):
	p1, p2 = line.p1, line.p2
	angle = atan2(p2[1]-p1[1], p2[0]-p1[0])
	# 把角度調回 0 到 pi 之間
	if angle < 0: return angle+pi
	elif angle == pi: return 0.0
	else: return angle


def rotate(a, rad): # 向量 a 旋轉 rad 角度
	return [a[0]*cos(rad)-a[1]*sin(rad), a[0]*sin(rad)+a[1]*cos(rad)]

def relationLineLine(a,b):
	if cross(sub(a.p2,a.p1), sub(b.p2,b.p1))==0:
		if relationPointLine(a.p1, b)=="on": return "same"
		else: return "parallel"
	return "cross"

def crossLineLine(x,y):
	a,b,c,d = x.p1, x.p2, y.p1, y.p2
	s1 = cross(sub(b,a), sub(c,a))
	s2 = cross(sub(b,a), sub(d,a))
	return cmul(1.0/(s2-s1), [c[0]*s2-d[0]*s1, c[1]*s1-d[1]*s1])

def crossSegSeg(x,y):
	a,b,c,d = x.p1, x.p2, y.p1, y.p2
	c1 = cross(sub(b,a), sub(c,a))
	c2 = cross(sub(b,a), sub(d,a))
	d1 = cross(sub(d,c), sub(a,c))
	d2 = cross(sub(d,c), sub(b,c))
	return c1*c2<0 and d1*d2<0

# 檢測轉向：結果為正代表是順時針右轉 (負代表逆時針左轉)
def direction(p0, p1, p2):
	return cross(sub(p2, p0), sub(p1, p0))

# 檢測 pk 是否位於 pi, pj 所形成的矩形內部。
def inBox(pi, pj, pk):
	return (min(pi[0], pj[0]) <= pk[0] and pk[0] <= max(pi[0], pj[0]) and
			min(pi[1], pj[1]) <= pk[1] and pk[1] <= max(pi[1], pj[1]))

# 檢測 (p1, p2) 和 (p3, p4) 是否相交 
def intersect(p1, p2, p3, p4):
	d1 = direction(p3, p4, p1)
	d2 = direction(p3, p4, p2)
	d3 = direction(p1, p2, p3)
	d4 = direction(p1, p2, p4)
	if d1*d2 < 0 and d3*d4 < 0: return True
	if d1==0 and inBox(p3, p4, p1): return True
	if d2==0 and inBox(p3, p4, p2): return True
	if d3==0 and inBox(p1, p2, p3): return True
	if d4==0 and inBox(p1, p2, p4): return True
	return False

def relationPointLine(point, line): # 點在線的哪一邊
	p, p1, p2 = point, line.p1, line.p2
	c = cross(sub(p, p1), sub(p2, p1))
	if (c<0): return "left"
	elif (c>0): return "right"
	else: return "on"

class Circle: # 圓
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

def relationSegmentCircle(segment, circle): # 線段與圓的關係
	s, c = segment, circle
	dst = distancePointSegment(c.center, s)
	r = dst - c.radius
	if r<0: return "in" # 線段有部分在圓內
	elif r==0: return "cut" # 線段和圓相切
	else: return "out" # 線段完全在圓外部

def relationLineCircle(line, circle): # 直線與圓的關係
	l, c = segment, circle
	dst = distancePointSegment(c.center, l)
	r = sub(dst, c)
	if r<0: return "in" # 線有部分在圓內
	elif r==0: return "cut" # 線和圓相切
	else: return "out" # 線完全在圓外部

def lineCrossCircle(line, circle):
	p1, p2, c, r = line.p1, line.p2, circle.center, circle.radius
	if relationLineCircle(line, circle)=="out": return [] # 線在圓外，沒有交點
	q = projectionPointLine(c, line) # 圓心在直線上的投影點
	d = distancePointLine(c, line) # 圓心到直線的距離
	k = sqrt(r**2-d**2)
	if k == 0: return [q] # 相切，只有一個交點
	# 否則， k > 0，有交點
	v = sub(p2, p1)
	n = v/length(v) # 線段上的單位向量
	kn = cmul(k, n)
	return [add(q, kn), sub(q, kn)] # 焦點 = 投影點+-k倍單位向量

def polygonArea(p): # p: points
	area = 0
	n = len(p)
	for i in range(n):
		area += cross(p[i], p[(i+1)%n]) # cross = 2*area
	return area/2.0

def polygonCenter(p): # p: points
	n = len(p)
	c = [0.0]*n
	if polygonArea(p)==0.0: return c
	for i in range(0, n):
		p1, p2 = p[i], p[(i+1)%n]
		c = add(c, cmul(cross(p1, p2), add(p1, p2)))
	return c/(polygonArea(p)*6.0)

if __name__ == '__main__':
	print('cross([1,2], [2,1])=', cross([1.0,2.0], [2.0,1.0]))
	p0 = [0.0,0.0]; p1 = [1.0,1.0]; p2 = [2.0, 1.0]; p3 = [-1.0, 0.0]
	print('direction(p0, p1, p2)=', direction(p0, p1, p2))
	print('intersect(p0, p1, p2, p3)=', intersect(p0, p1, p2, p3))
	l1 = Line(p0, p1)
	print('relationPointLine(p2, l1)=', relationPointLine(p2, l1))

