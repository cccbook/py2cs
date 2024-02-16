class Ratio:
    def __init__(self,a,b): self.a,self.b = a,b  
    
    def mul(self,r2) : return Ratio(self.a*r2.a, self.b*r2.b)  
    
    def div(self,r2) : return Ratio(self.a*r2.b, self.b*r2.a)  
    
    def inv(self) : return Ratio(self.b, self.a)  
    
    def add(self,r2) : return Ratio(self.a*r2.b+self.b*r2.a, self.b*r2.b)
    
    def sub(self,r2) : return Ratio(self.a*r2.b-self.b*r2.a, self.b*r2.b)
    
    def __str__(self): return f'{self.a}/{self.b}'  

r1 = Ratio(2,3) 
print(r1) 

r2 = r1.mul(r1).add(r1) 
print(r2)
