# http

## client

最簡單的請求！

```
GET /aaa/bbb/ccc.html HTTP/1.1
空行

```

## server

```
HTTP/1.1 200 OK
Content-Length: 12
空行
Hello World!
```

## curl

```
mac020:ccc mac020$ curl -v http://localhost:8080
* Rebuilt URL to: http://localhost:8080/
*   Trying ::1...
* TCP_NODELAY set
* Connection failed
* connect to ::1 port 8080 failed: Connection refused
*   Trying fe80::1...
* TCP_NODELAY set
* Connection failed
* connect to fe80::1 port 8080 failed: Connection refused
*   Trying 127.0.0.1...
* TCP_NODELAY set
* Connected to localhost (127.0.0.1) port 8080 (#0)
> GET / HTTP/1.1
> Host: localhost:8080
> User-Agent: curl/7.54.0
> Accept: */*
> 
< HTTP/1.1 200 OK
< Content-Type: text/plain; charset=UTF-8
< Content-Length: 14
< 
Hello World!
* Connection #0 to host localhost left intact
```