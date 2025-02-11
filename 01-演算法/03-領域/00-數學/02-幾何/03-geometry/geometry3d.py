from vector import *
from math import *

def cross(a,b): # (外積) 叉積 a0*b1-a1*b0 (只適用於二維向量)
	return 

def normal(a): # 向量 a 的法向量
	return 

def rotate(a, rad): # 向量 a 旋轉 rad 角度
	return 

def pointOnPlane(p, plane): # 點 p 是否在平面 plane 上
    pass

def parallel(plane1, plane2): # 兩平面是否平行

def vertical(plane1, plane2): # 兩平面是否垂直

def lineCrossPlane(line, plane): # 直線與平面的交點

def volume4(a,b,c,d): # a,b,c,d 形成的四面體體積
    return dot(cross(sub(b,a), sub(c,a)), sub(d,a))/6.0

if __name__ == '__main__':
