

## run

```
fastapi dev main.py
```

## test


```
(base) teacher@teacherdeiMac 05-blog % pytest
=================== test session starts ====================
platform darwin -- Python 3.11.7, pytest-7.4.0, pluggy-1.0.0
rootdir: /Users/teacher/Desktop/ccc/py2cs/09-應用程式/08-fastapi/05-blog
plugins: anyio-4.2.0
collected 3 items                                          

test_main.py ...                                     [100%]

===================== warnings summary =====================
database.py:10
  /Users/teacher/Desktop/ccc/py2cs/09-應用程式/08-fastapi/05-blog/database.py:10: MovedIn20Warning: The ``declarative_base()`` function is now available as sqlalchemy.orm.declarative_base(). (deprecated since: 2.0) (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)
    Base = declarative_base()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=============== 3 passed, 1 warning in 2.18s ===============
```