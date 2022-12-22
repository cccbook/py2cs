from jinja2 import Template
template = Template('Hello {{ name }}!')
text = template.render(name='John Doe')
print('text=', text)
