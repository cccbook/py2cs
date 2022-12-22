import json
text = json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
print(f'text={text}')
obj = json.loads(text)
print(f'obj={obj}')
