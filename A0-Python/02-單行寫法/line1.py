fruit_list = ['apple', 'orange', 'banana', 'kiwi']
my_fruit = ["test %s" % fruit for fruit in fruit_list]
print(my_fruit) # ['test apple', 'test orange', 'test banana', 'test kiwi'

number_list = [3, 4, 2, 3, 4.5, 8, 9, 8.2, 9.1, 10, 1]
get_number = [num for num in number_list if num > 8]
print(get_number) # [9, 8.2, 9.1, 10]
