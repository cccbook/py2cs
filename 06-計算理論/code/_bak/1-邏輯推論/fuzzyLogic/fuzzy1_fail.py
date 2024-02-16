from fuzzy.classes import Domain
from fuzzy.functions import R, S, alpha

T = Domain("test", 0, 30, res=0.1)

T.up = R(0,10)
T.top = alpha(floor=0.5, ceiling=1, func=S(0, 30))
T.down = S(20, 30)
T.polygon = up & top & down

T.polygon.plot()
T.not_polygon = ~T.polygon
T.not_polygon.plot()