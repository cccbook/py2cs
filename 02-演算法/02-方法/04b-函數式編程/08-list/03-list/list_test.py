from pair import *
from list import *

one_through_four = list_new(1, 2, 3, 4)
print('head(one_through_four)=', head(one_through_four))
print('list_ref(one_through_four,2)=', list_ref(one_through_four, 2))
list_print(one_through_four)
ten = map(lambda x:x*10, one_through_four)
list_print(ten)
foreach(lambda x:print(x), one_through_four)
