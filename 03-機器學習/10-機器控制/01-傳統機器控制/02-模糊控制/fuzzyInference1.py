from fuzzylogic.classes import Domain, Set, Rule
from fuzzylogic.hedges import very
from fuzzylogic.functions import R, S

temp = Domain("Temperature", -80, 80)
hum = Domain("Humidity", 0, 100)
motor = Domain("Speed", 0, 2000)

temp.cold = S(0,20)
temp.hot = R(15,30)

hum.dry = S(20,50)
hum.wet = R(40,70)

motor.fast = R(1000,1500)
motor.slow = ~motor.fast

R1 = Rule({(temp.hot, hum.dry): motor.fast})
R2 = Rule({(temp.cold, hum.dry): very(motor.slow)})
R3 = Rule({(temp.hot, hum.wet): very(motor.fast)})
R4 = Rule({(temp.cold, hum.wet): motor.slow})

rules = Rule({(temp.hot, hum.dry): motor.fast,
              (temp.cold, hum.dry): very(motor.slow),
              (temp.hot, hum.wet): very(motor.fast),
              (temp.cold, hum.wet): motor.slow,
             })

rules == R1 | R2 | R3 | R4 == sum([R1, R2, R3, R4])

values = {hum: 45, temp: 22}
print(R1(values), R2(values), R3(values), R4(values), "=>", rules(values))