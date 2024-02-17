import json

obj = {'name':'ccc', 'age':54}
print(obj)
print(json.dumps(obj))
print(json.dumps(obj, indent=2))

