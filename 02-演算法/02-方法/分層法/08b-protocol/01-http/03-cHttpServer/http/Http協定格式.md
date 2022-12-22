# http 協定格式

## 基本格式

請求 request 與回應 response，表頭 header 與內容 body 之間都是用兩個換行 \r\n\r\n (一個空行) 所隔開。 

* [Notes on the format of HTTP requests and responses](https://docs.citrix.com/en-us/citrix-adc/current-release/appexpert/http-callout/http-request-response-notes-format.html)
* https://www.w3.org/TR/html401/interact/forms.html

## Post 類型的 request 之 body

body 的格式有五種 

1. raw : 不編碼，直接傳遞原始格式
2. x-www-form-urlencoded: 一般 form 使用的 格式
3. form-data: 有附加檔案時，會採用者種稱為 multi-part 的 form-data 格式
4. binary : 二進位格式，例如串流影音可用
5. GraphQL : GraphQL 專用格式

* [Postman Chrome: What is the difference between form-data, x-www-form-urlencoded and raw](https://stackoverflow.com/questions/26723467/postman-chrome-what-is-the-difference-between-form-data-x-www-form-urlencoded)

## 格式範例

1. raw : 不編碼，直接附上去

```
POST /jsondb/insert HTTP/1.1     
Content-Type: text/plain
User-Agent: PostmanRuntime/7.29.2
Accept: */*
Postman-Token: c824ffb2-b0aa-4966-a7a3-08729780f9a1
Host: 127.0.0.1:8080
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Content-Length: 22

{"name":"ccc", age:53}
op=POST path=/jsondb/insert body={"name":"ccc", age:53}
```

2. x-www-form-urlencoded: 一般 form 使用的 格式

```
POST /jsondb/ HTTP/1.1
User-Agent: PostmanRuntime/7.29.2
Accept: */*
Postman-Token: 3e65c2be-b59f-4b48-be4d-53910ed7d42d
Host: 127.0.0.1:8080
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Content-Type: application/x-www-form-urlencoded
Content-Length: 35

insert=%7B%22name%22%3A%22ccc%22%7D
```

3. form-data

有附加檔案時，會採用者種稱為 multi-part 的 form-data 格式

這種格式會產生一個內容中沒出現過的隨機分隔字串，用來做為 multi-part 之間的區分。

```
POST /jsondb/insert HTTP/1.1
User-Agent: PostmanRuntime/7.29.2
Accept: */*
Postman-Token: 1cf926e3-640c-4d6a-8400-1aeac3460b2c
Host: 127.0.0.1:8080
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Content-Type: multipart/form-data; boundary=--------------------------519277740013837479938406
Content-Length: 266

----------------------------519277740013837479938406
Content-Disposition: form-data; name="name"

ccc
----------------------------519277740013837479938406
Content-Disposition: form-data; name="age"

53
----------------------------519277740013837479938406--
```

4. binary : Binary can be used when you want to attach non-textual data to the request, e.g. a video/audio file, images, or any other binary data file.

若我將 Makefile 的內容附加上去，則會得到

```
POST /jsondb/insert HTTP/1.1   
User-Agent: PostmanRuntime/7.29.2
Accept: */*
Postman-Token: a0bb97a7-1dd2-46f4-a1a4-6437e3ecaff2
Host: 127.0.0.1:8080
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Content-Length: 100
Content-Type: application/x-sh

gcc -Wall -D_DEBUG docDB.c dbBuild1.c -o dbBuild1
gcc -Wall -D_DEBUG docDB.c dbQuery1.c -o dbQuery1
op=POST path=/jsondb/insert body=gcc -Wall -D_DEBUG docDB.c dbBuild1.c -o dbBuild1
gcc -Wall -D_DEBUG docDB.c dbQuery1.c -o dbQuery1
```

5. GraphQL : GraphQL 專用格式

若傳入的 graphql 如下

```
{
  hero {
    name
    # Queries can have comments!
    friends {
      name
    }
  }
}
```

則應傳出下列請求給 server

```
===========request=============
POST /jsondb/insert HTTP/1.1
Content-Type: application/json
User-Agent: PostmanRuntime/7.29.2
Accept: */*
Postman-Token: 6c207be4-8bfc-4d54-9a12-84b810b4f92c
Host: 127.0.0.1:8080
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Content-Length: 125

{"query":"{\r\n  hero {\r\n    name\r\n    # Queries can have comments!\r\n  
  friends {\r\n      name\r\n    }\r\n  }\r\n}"}
op=POST path=/jsondb/insert body={"query":"{\r\n  hero {\r\n    name\r\n    # Queries can have comments!\r\n    friends {\r\n      name\r\n    }\r\n  }\r\n}"}
```