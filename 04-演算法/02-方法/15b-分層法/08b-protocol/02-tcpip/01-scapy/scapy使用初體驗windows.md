# Scapy

```
Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/alg/03b-protocol/02-tc/02-tcp/01-scapy (master)
$ scapy -H
WARNING: No libpcap provider available ! pcap won't be used  used
Welcome to Scapy (2.5.0rc2) using IPython 8.6.0       
>>> a=IP(ttl=10)
>>> a
<IP  ttl=10 |>
>>> a.src
WARNING: No route found (no default route?)
'0.0.0.0'
>>> a.dst="192.168.1.1"
>>> a
<IP  ttl=10 dst=192.168.1.1 |>
>>> a.src
WARNING: No route found (no default route?)
'0.0.0.0'
>>> del(a.ttl)
>>> a
<IP  dst=192.168.1.1 |>
>>> a.ttl
64
>>> IP
scapy.layers.inet.IP
>>> IP()/TCP()
<IP  frag=0 proto=tcp |<TCP  |>>
>>> Ether()/IP()/TCP()
<Ether  type=IPv4 |<IP  frag=0 proto=tcp |<TCP  |>>>
>>> IP()/TCP()/"GET / HTTP/1.0\r\n\r\n"
<IP  frag=0 proto=tcp |<TCP  |<Raw  load='GET / HTTP/1.0\r\n\r\n' |>>>
>>> Ether()/IP()/IP()/UDP()
<Ether  type=IPv4 |<IP  frag=0 proto=4 |<IP  frag=0 proto=udp |<UDP  |>>>>
>>> IP(proto=55)/TCP()
<IP  frag=0 proto=55 |<TCP  |>>
>>> raw(IP())
WARNING: No route found (no default route?)
b'E\x00\x00\x14\x00\x01\x00\x00@\x00\xfb\xe8\x00\x00\x00\x00\x7f\x00\x00\x01'
>>> IP(_)
<IP  version=4 ihl=5 tos=0x0 len=20 id=1 flags= frag=0 ttl=64 proto=ip chksum=0xfbe8 src=0.0.0.0 dst=127.0.0.1 |>
>>> a=Ether()/IP(dst="www.slashdot.org")/TCP()/"GET / 
...: index.html HTTP/1.0 \n\n"
>>> hexdump(a)
WARNING: No route found (no default route?)
WARNING: No route found (no default route?)
WARNING: more No route found (no default route?)
0000  FF FF FF FF FF FF 00 00 00 00 00 00 08 00 45 00 
 ..............E.
0010  00 43 00 01 00 00 40 06 F6 4C 00 00 00 00 68 12 
 .C....@..L....h.
0020  1C 56 00 14 00 50 00 00 00 00 00 00 00 00 50 02  .V...P........P.
0030  20 00 39 4A 00 00 47 45 54 20 2F 69 6E 64 65 78   .9J..GET /index
0040  2E 68 74 6D 6C 20 48 54 54 50 2F 31 2E 30 20 0A  .html HTTP/1.0 .
0050  0A                                 
>>> b=raw(a)
WARNING: No route found (no default route?)
WARNING: No route found (no default route?)
WARNING: more No route found (no default route?)
>>> b
b'\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x08\x00E\x00\x00C\x00\x01\x00\x00@\x06\xf6L\x00\x00\x00\x00h\x12\x1cV\x00\x14\x00P\x00\x00\x00\x00\x00\x00\x00\x00P\x02 \x009J\x00\x00GET /index.html HTTP/1.0 \n\n'
>>> c=Ether(b)
>>> c
<Ether  dst=ff:ff:ff:ff:ff:ff src=00:00:00:00:00:00 type=IPv4 |<IP  version=4 ihl=5 tos=0x0 len=67 id=1 flags= frag=0 ttl=64 proto=tcp chksum=0xf64c src=0.0.0.0 dst=104.18.28.86 |<TCP  sport=ftp_data dport=http seq=0 ack=0 dataofs=5 reserved=0 flags=S window=8192 chksum=0x394a urgptr=0 |<Raw  load='GET /index.html HTTP/1.0 \n\n' |>>>>
>>> c.hide_defaults()
>>> c
<Ether  dst=ff:ff:ff:ff:ff:ff src=00:00:00:00:00:00 type=IPv4 |<IP  ihl=5 len=67 frag=0 proto=tcp chksum=0xf64c src=0.0.0.0 dst=104.18.28.86 |<TCP  dataofs=5 chksum=0x394a |<Raw  load='GET /index.html HTTP/1.0 \n\n' |>>>>
>>> a=rdpcap("./isakmp.cap")
------------------------------------------------------------FileNotFoundError          Traceback (most recent call last)File ~\AppData\Local\Programs\Python\Python310\lib\site-packages\scapy\utils.py:1199, in PcapReader_metaclass.open(fname)
   1198 try:
-> 1199     fdesc = gzip.open(filename, "rb")  # type: _ByteStream
   1200     magic = fdesc.read(4)
```
