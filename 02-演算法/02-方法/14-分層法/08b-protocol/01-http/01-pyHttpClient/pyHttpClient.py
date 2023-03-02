import socket
HOST = 'example.com'    # The remote host
PORT = 80              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.send(b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n')
data = s.recv(65536)
s.close()
print('Received', repr(data))
