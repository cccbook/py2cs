# 來源 -- https://github.com/chwang12341/Learn-Python-/blob/master/simplehttpserver_learning/httpserver.py
# -*- coding: utf-8 -*-
import sys
import http.server
from http.server import SimpleHTTPRequestHandler

HandlerClass = SimpleHTTPRequestHandler
ServerClass  = http.server.HTTPServer
Protocol     = "HTTP/1.0"

if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 8000
server_address = ('127.0.0.1', port)

HandlerClass.protocol_version = Protocol
httpd = ServerClass(server_address, HandlerClass)

sa = httpd.socket.getsockname()
print(f"Serving HTTP on http://{sa[0]}:{sa[1]}")
httpd.serve_forever()
