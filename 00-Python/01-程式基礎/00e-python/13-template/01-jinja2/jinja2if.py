from jinja2 import Template
temp = '''
{% if user %}
  Hello, {{ user }}
{% else %}
  Hello, Stranger
{% endif %}
'''
template = Template(temp)
# text = template.render(user='John')
text = template.render(user=None)
print('text=', text)