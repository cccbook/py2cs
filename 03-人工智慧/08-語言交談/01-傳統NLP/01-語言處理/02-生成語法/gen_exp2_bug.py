import random as r

'''
E = N | E [+/-*] E
N = 0-9 | (E)
'''

def E():
	gen = r.choice(["N", "EE"])
	# print('gen=', gen)
	if gen == "N":
		return N()
	else:
		return E() + r.choice(["+", "-", "*", "/"]) + E()

def N():
    rnd = r.random()
    if rnd > 0.2:
	    return r.choice(["1", "2", "3", "4", "5", "6", "7", "8", "9"])
    else:
        return "("+E()+")"

e = E()
print(e, "=", eval(e))
