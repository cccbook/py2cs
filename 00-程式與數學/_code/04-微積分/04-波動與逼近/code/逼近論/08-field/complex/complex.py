import math

class Complex :
    def __init__(self,a,b):
        self.a,self.b = a,b

    def conj(self):
        return Complex(self.a, -1*self.b)  
    
    def add(self, c2):
        return Complex(self.a+c2.a, self.b+c2.b)  
    
    def sub(self,c2):
        return Complex(self.a-c2.a, self.b-c2.b)  
    
    def mul(self,c2):
        a,b = self.a,self.b
        c,d = c2.a,c2.b 
        return Complex(a*c-b*d, a*d+b*c) 
    
    def div(self,c2):
        a,b = self.a,self.b
        c,d = c2.a,c2.b 
        return Complex((a*c+b*d)/(c*c+d*d), (b*c-a*d)/(c*c+d*d)) 
    
    def __str__(self):
        return f'{self.a}+{self.b}i'  

    def ln(self):
        a,b = self.a,self.b
        r=a*a+b*b 
        w = 1/2*math.log(r) 
        x = math.acos(a/math.sqrt(r)) 
        return Complex(w, x) 
    
    def exp(self):
        a,b = self.a,self.b 
        r=math.exp(a) 
        return Complex(r*math.cos(b), r*math.sin(b)) 

c1 = Complex(2,3) 
print(c1) 

c2 = c1.add(c1).mul(c1).div(c1) 
print(c2) 

print(c1.ln()) 

print(c1.exp().ln())
