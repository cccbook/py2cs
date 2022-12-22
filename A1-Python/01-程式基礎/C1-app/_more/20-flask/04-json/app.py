from flask import Flask
from flask import jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Hello, World!</h1>"

@app.route('/user/<name>')
def user(name):
    d = {'name':name}
    return jsonify(d)
    # return '<h1>Hello, {0}!</h1>'.format(name)

if __name__ == '__main__':
    app.run(debug=True)

