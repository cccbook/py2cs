add_xyz_lambda = lambda x,y,z: x+y+z

print(f'add_xyz_lambda(1,2,3)={add_xyz_lambda(1,2,3)}')

add_xyz_curry = lambda x:lambda y:lambda z:x+y+z

print(f'add_xyz_curry(1)(2)(3)={add_xyz_curry(1)(2)(3)}')

def add_xyz(x,y,z):
    def addx_yz(y,z):
        def addxy_z(z):
            return x+y+z
        return addxy_z(z)
    return addx_yz(y,z)
    
print(f'add_xyz(1,2,3)={add_xyz(1,2,3)}')

