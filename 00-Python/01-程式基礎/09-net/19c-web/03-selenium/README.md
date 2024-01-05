# Selenium

* [How to automate opening and login to websites with Python](https://medium.com/@kikigulab/how-to-automate-opening-and-login-to-websites-with-python-6aeaf1f6ae98)

## 下載 Chrome Driver

* https://sites.google.com/chromium.org/driver/

## 更新瀏覽器

* 更新方法 -- https://support.google.com/chrome/answer/95414?hl=en&co=GENIE.Platform%3DDesktop

如果瀏覽器不夠新，可能會有下列錯誤

```
selenium.common.exceptions.SessionNotCreatedException: Message: session not created: This version of ChromeDriver only supports Chrome version 107
Current browser version is 106.0.5249.119 with binary path C:\Program Files (x86)\Google\Chrome\Application\chrome.exe
```

## 安裝 Selenium

```
$ pip install Selenium
Collecting Selenium
  Downloading selenium-4.5.0-py3-none-any.whl (995 kB)
     ------------------------------------ 995.2/995.2 kB 2.3 MB/s eta 0:00:00 
Requirement already satisfied: urllib3[socks]~=1.26 in c:\users\hero3c\appdata\local\programs\python\python311\lib\site-packages (from Selenium) (1.26.12)  
Collecting trio~=0.17
  Downloading trio-0.22.0-py3-none-any.whl (384 kB)
     ------------------------------------ 384.9/384.9 kB 1.5 MB/s eta 0:00:00 
Collecting trio-websocket~=0.9
  Downloading trio_websocket-0.9.2-py3-none-any.whl (16 kB)
Requirement already satisfied: certifi>=2021.10.8 in c:\users\hero3c\appdata\local\programs\python\python311\lib\site-packages (from Selenium) (2022.9.24)  
Collecting attrs>=19.2.0
  Using cached attrs-22.1.0-py2.py3-none-any.whl (58 kB)
Requirement already satisfied: sortedcontainers in c:\users\hero3c\appdata\local\programs\python\python311\lib\site-packages (from trio~=0.17->Selenium) (2.3.0)
Collecting async-generator>=1.9
  Downloading async_generator-1.10-py3-none-any.whl (18 kB)
Requirement already satisfied: idna in c:\users\hero3c\appdata\local\programs\python\python311\lib\site-packages (from trio~=0.17->Selenium) (3.4)
Collecting outcome
  Downloading outcome-1.2.0-py2.py3-none-any.whl (9.7 kB)
Requirement already satisfied: sniffio in c:\users\hero3c\appdata\local\programs\python\python311\lib\site-packages (from trio~=0.17->Selenium) (1.3.0)     
Collecting cffi>=1.14
  Downloading cffi-1.15.1-cp311-cp311-win_amd64.whl (179 kB)
     ------------------------------------ 179.0/179.0 kB 5.4 MB/s eta 0:00:00
Collecting wsproto>=0.14
  Downloading wsproto-1.2.0-py3-none-any.whl (24 kB)
Collecting PySocks!=1.5.7,<2.0,>=1.5.6
  Downloading PySocks-1.7.1-py3-none-any.whl (16 kB)
Collecting pycparser
  Using cached pycparser-2.21-py2.py3-none-any.whl (118 kB)
Requirement already satisfied: h11<1,>=0.9.0 in c:\users\hero3c\appdata\local\programs\python\python311\lib\site-packages (from wsproto>=0.14->trio-websocket~=0.9->Selenium) (0.12.0)
Installing collected packages: wsproto, PySocks, pycparser, attrs, async-generator, outcome, cffi, trio, trio-websocket, Selenium
Successfully installed PySocks-1.7.1 Selenium-4.5.0 async-generator-1.10 attrs-22.1.0 cffi-1.15.1 outcome-1.2.0 pycparser-2.21 trio-0.22.0 trio-websocket-0.9.2 wsproto-1.2.0
```