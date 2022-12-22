

from jinja2 import Template
temp = '''
<ul>
{% for user in users %}
  <li><a href="{{ user.url }}">{{ user.username }}</a></li>
{% endfor %}
</ul>
'''
template = Template(temp)
html = template.render(users=[
  {'url':'http://people.com/John', 'username':'John'},
  {'url':'http://people.com/Peter', 'username':'Peter'},
])
print('html=', html)