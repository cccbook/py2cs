# Flask

* https://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart


## hello.py

```
user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109a/ws/_more/python/flask (master)
$ pip install flask
Collecting flask
  Downloading https://files.pythonhosted.org/packages/f2/28/2a03252dfb9ebf377f40fba6a7841b47083260bf8bd8e737b0c6952df83f/Flask-1.1.2-py2.py3-none-any.whl (94kB)
     |████████████████████████████████| 102kB 284kB/s
Requirement already satisfied: Jinja2>=2.10.1 in c:\users\user\appdata\roaming\python\python37\site-packages (from flask) (2.10.1)
Requirement already satisfied: Werkzeug>=0.15 in c:\users\user\appdata\local\programs\python\python37\lib\site-packages (from flask) (0.16.0)
Collecting itsdangerous>=0.24
  Downloading https://files.pythonhosted.org/packages/76/ae/44b03b253d6fade317f32c24d100b3b35c2239807046a4c953c7b89fa49e/itsdangerous-1.1.0-py2.py3-none-any.whl
Collecting click>=5.1
  Downloading https://files.pythonhosted.org/packages/d2/3d/fa76db83bf75c4f8d338c2fd15c8d33fdd7ad23a9b5e57eb6c5de26b430e/click-7.1.2-py2.py3-none-any.whl (82kB)
     |████████████████████████████████| 92kB 280kB/s
Requirement already satisfied: MarkupSafe>=0.23 in c:\users\user\appdata\roaming\python\python37\site-packages (from Jinja2>=2.10.1->flask) (1.1.1)
Installing collected packages: itsdangerous, click, flask
Successfully installed click-7.1.2 flask-1.1.2 itsdangerous-1.1.0
WARNING: You are using pip version 19.3.1; however, version 20.1.1 is available.
You should consider upgrading via the 'python -m pip install --upgrade pip' command.

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109a/ws/_more/python/flask (master)
$ $env:FLASK_APP = "hello.py"
bash: :FLASK_APP: command not found

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109a/ws/_more/python/flask (master)
$ set FLASK_APP=hello.py

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109a/ws/_more/python/flask (master)
$ export FLASK_APP=hello.py

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109a/ws/_more/python/flask (master)
$ python -m flask run
 * Serving Flask app "hello.py"
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
127.0.0.1 - - [16/Jul/2020 14:14:53] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [16/Jul/2020 14:14:54] "GET /favicon.ico HTTP/1.1" 404 -

```

## route.py

```
user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109a/ws/_more/python/flask (master)
$ export FLASK_APP=route.py

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109a/ws/_more/python/flask (master)
$ python -m flask run
 * Serving Flask app "route.py"
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
/
/login
/login?next=%2F
/user/John%20Doe
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
127.0.0.1 - - [16/Jul/2020 14:17:29] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [16/Jul/2020 14:17:42] "GET /login HTTP/1.1" 200 -
127.0.0.1 - - [16/Jul/2020 14:17:57] "GET /user/snoopy HTTP/1.1" 200 -
```