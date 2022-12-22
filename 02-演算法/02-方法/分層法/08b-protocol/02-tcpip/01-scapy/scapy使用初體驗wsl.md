# scapy 使用初體驗 wsl

## 以下方式會有權限不足的情形

```
$ sudo apt-get update
$ sudo apt-get -y install python3-pip

$ sudo apt-get install libpcap-dev
$ pip install --pre scapy[basic]
```

改用

```
$ sudo apt-get install scapy
$ sudo scapy -H
```

## 使用

```
ccc@DESKTOP-O093POU:/mnt/c/ccc/alg/03b-protocol/02-tcp/02cket$ scapy -H
Welcome to Scapy (2.5.0rc2) using IPython 8.6.0
>>>
>>> a=IP(ttl=10)
>>> a
<IP  ttl=10 |>
>>> a.src
'127.0.0.1'
>>> a.dst
'127.0.0.1'
>>> a
<IP  ttl=10 |>
>>> del(a.ttl)
>>> a
<IP  |>
>>> a.ttl
64
>>> IP
scapy.layers.inet.IP
>>> IP()
<IP  |>
>>> IP()/TCP()
<IP  frag=0 proto=tcp |<TCP  |>>
>>> Ether()/IP()/TCP()
<Ether  type=IPv4 |<IP  frag=0 proto=tcp |<TCP  |>>>
>>> IP()/TCP()/"GET / HTTP/1.0\r\n\r\n"
<IP  frag=0 proto=tcp |<TCP  |<Raw  load='GET / HTTP/1.0\\r\n' |>>>
>>> Ether()/IP()/IP()/UDP()
<Ether  type=IPv4 |<IP  frag=0 proto=ipencap |<IP  frag=0>>> IP(proto=55)/TCP()
<IP  frag=0 proto=55 |<TCP  |>>
>>> raw(IP())
b'E\x00\x00\x14\x00\x01\x00\x00@\x00|\xe7\x7f\x00\x00\x001\x7f\x00\x00\x01'                                     1
>>> IP(_)
<IP  version=4 ihl=5 tos=0x0 len=20 id=1 flags= frag=0 t
ttl=64 proto=hopopt chksum=0x7ce7 src=127.0.0.1 dst=1270t.0.0.1 |>                                                
>>> >>> a=Ether()/IP(dst="www.slashdot.org")/TCP()/"GET e />>> a=Ether()/IP(dst="www.slashdot.org")/TCP()/"GET /d
>>> a=Ether()/IP(dst="www.slashdot.org")/TCP()/"GET /in...: dex.html HTTP/1.0 \n\n"
>>> hexdump(a)
WARNING: getmacbyip failed on [Errno 1] Operation not permitted
WARNING: Mac address to reach destination not found. Using broadcast.
0000  FF FF FF FF FF FF 00 15 5D 15 BF 35 08 00 45 00  
........]..5..E.
0010  00 43 00 01 00 00 40 06 8D A6 AC 18 BC 8D 68 12  
 .C....@.......h.
0020  1C 56 00 14 00 50 00 00 00 00 00 00 00 00 50 02  
 .V...P........P.
0030  20 00 D0 A3 00 00 47 45 54 20 2F 69 6E 64 65 78  
  .....GET /index
0040  2E 68 74 6D 6C 20 48 54 54 50 2F 31 2E 30 20 0A  
 .html HTTP/1.0 .
0050  0A
 .
>>> b = raw(a)
WARNING: getmacbyip failed on [Errno 1] Operation not 
permitted
WARNING: Mac address to reach destination not found. Using broadcast.
>>> b
b'\xff\xff\xff\xff\xff\xff\x00\x15]\x15\xbf5\x08\x00E\x00\x00C\x00\x01\x00\x00@\x06\x8d\xa6\xac\x18\xbc\x8dh\x12\x1cV\x00\x14\x00P\x00\x00\x00\x00\x00\x00\x00\x00P\x02 \x00\xd0\xa3\x00\x00GET /index.html HTTP/1.0 \n\n'
>>> c=Ether(b)
>>> c
<Ether  dst=ff:ff:ff:ff:ff:ff src=00:15:5d:15:bf:35 type=IPv4 |<IP  version=4 ihl=5 tos=0x0 len=67 id=1 flags= frag=0 ttl=64 proto=tcp chksum=0x8da6 src=172.24.188.141 dst=104.18.28.86 |<TCP  sport=ftp_data dport=http seq=0 ack=0 dataofs=5 reserved=0 flags=S window=8192 chksum=0xd0a3 urgptr=0 |<Raw  load='GET /index.html HTTP/1.0 \n\n' |>>>>
>>> c.hide_defaults()
>>> c
<Ether  dst=ff:ff:ff:ff:ff:ff src=00:15:5d:15:bf:35 type=IPv4 |<IP  ihl=5 len=67 frag=0 proto=tcp chksum=0x8da6 src=172.24.188.141 dst=104.18.28.86 |<TCP  dataofs=5 chksum=0xd0a3 |<Raw  load='GET /index.html HTTP/1.0 \n\n' |>>>>
>>> a=rdpcap("/spare/captures/isakmp.cap")
------------------------------------------------------FileNotFoundError    Traceback (most recent call last)File ~/.local/lib/python3.8/site-packages/scapy/utils.py:1199, in PcapReader_metaclass.open(fname)
   1198 try:
-> 1199     fdesc = gzip.open(filename, "rb")  # type: _ByteStream
   1200     magic = fdesc.read(4)
```

## Generating sets of packets

```
>>> a=IP(dst="www.slashdot.org/30")        
>>> a
<IP  dst=Net("www.slashdot.org/30") |>
>>> [p for p in a]
[<IP  dst=104.18.28.84 |>,
 <IP  dst=104.18.28.85 |>,
 <IP  dst=104.18.28.86 |>,
 <IP  dst=104.18.28.87 |>]
>>> [p for p in a]
[<IP  dst=104.18.28.84 |>,
 <IP  dst=104.18.28.85 |>,
 <IP  dst=104.18.28.86 |>,
 <IP  dst=104.18.28.87 |>]
>>> b=IP(ttl=[1,2,(5,9)])
>>> b
<IP  ttl=[1, 2, (5, 9)] |>
>>> [p for p in b]
[<IP  ttl=1 |>,
 <IP  ttl=2 |>,
 <IP  ttl=5 |>,
 <IP  ttl=6 |>,
 <IP  ttl=7 |>,
 <IP  ttl=8 |>,
 <IP  ttl=9 |>]
>>> c=TCP(dport=[80,443])
>>> [p for p in a/c]
[<IP  frag=0 proto=tcp dst=104.18.28.84 |<TCP  dport=http |>>,
 <IP  frag=0 proto=tcp dst=104.18.28.84 |<TCP  dport=https |>>,
 <IP  frag=0 proto=tcp dst=104.18.28.85 |<TCP  dport=http |>>,
 <IP  frag=0 proto=tcp dst=104.18.28.85 |<TCP  dport=https |>>,
 <IP  frag=0 proto=tcp dst=104.18.28.86 |<TCP  dport=http |>>,
 <IP  frag=0 proto=tcp dst=104.18.28.86 |<TCP  dport=https |>>,
 <IP  frag=0 proto=tcp dst=104.18.28.87 |<TCP  dport=http |>>,
 <IP  frag=0 proto=tcp dst=104.18.28.87 |<TCP  dport=https |>>]
>>> p = PacketList(a)
>>> p
<PacketList: TCP:0 UDP:0 ICMP:0 Other:4>
>>> p = PacketList([p for p in a/c])       
>>> p
<PacketList: TCP:8 UDP:0 ICMP:0 Other:0>
>>> send(IP(dst="1.2.3.4")/ICMP())
-------------------------------------------PermissionErrorTraceback (most recent call 
last)
Cell In [40], line 1
----> 1 send(IP(dst="1.2.3.4")/ICMP())     
```

## Send and receive packets (sr)

```
sudo scapy -H
INFO: Can't import matplotlib. Won't be able to plot.
INFO: Can't import PyX. Won't be able to use psdump() or pdfdump().
WARNING: No route found for IPv6 destination :: (no default route?)
Welcome to Scapy (2.4.3) using IPython 7.13.0
>>> send(IP(dst="1.2.3.4")/ICMP())
.
Sent 1 packets.
>>> sendp(Ether()/IP(dst="1.2.3.4",ttl=(1,4)), iface="eth1")      
------------------------------------------------------------------OSError                          Traceback (most recent call last)<ipython-input-2-b630fd4d00bd> in <module>
----> 1 sendp(Ether()/IP(dst="1.2.3.4",ttl=(1,4)), iface="eth1")  

/usr/lib/python3/dist-packages/scapy/sendrecv.py in sendp(x, inter, loop, iface, iface_hint, count, verbose, realtime, return_packets, socket, *args, **kargs)
    333         iface = conf.route.route(iface_hint)[0]
    334     need_closing = socket is None
--> 335     socket = socket or conf.L2socket(iface=iface, *args, **kargs)
    336     results = __gen_send(socket, x, inter=inter, loop=loop,
    337                          count=count, verbose=verbose,    

/usr/lib/python3/dist-packages/scapy/arch/linux.py in __init__(self, iface, type, promisc, filter, nofilter, monitor)
    475                 attach_filter(self.ins, filter, iface)    
    476         if self.promisc:
--> 477             set_promisc(self.ins, self.iface)
    478         self.ins.bind((self.iface, type))
    479         _flush_fd(self.ins)

/usr/lib/python3/dist-packages/scapy/arch/linux.py in set_promisc(s, iff, val)
    163
    164 def set_promisc(s, iff, val=1):
--> 165     mreq = struct.pack("IHH8s", get_if_index(iff), PACKET_MR_PROMISC, 0, b"")
    166     if val:
    167         cmd = PACKET_ADD_MEMBERSHIP

/usr/lib/python3/dist-packages/scapy/arch/linux.py in get_if_index(iff)
    378
    379 def get_if_index(iff):
--> 380     return int(struct.unpack("I", get_if(iff, SIOCGIFINDEX)[16:20])[0])
    381
    382

/usr/lib/python3/dist-packages/scapy/arch/common.py in get_if(iff, cmd)
     57
     58     sck = socket.socket()
---> 59     ifreq = ioctl(sck, cmd, struct.pack("16s16x", iff.encode("utf8")))
     60     sck.close()
     61     return ifreq

OSError: [Errno 19] No such device
>>> p = sr1(IP(dst="www.slashdot.org")/ICMP()/"XXXXXXXXXXX")      
Begin emission:
Finished sending 1 packets.
*
Received 1 packets, got 1 answers, remaining 0 packets
>>> p
<IP  version=4 ihl=5 tos=0x0 len=39 id=7244 flags= frag=0 ttl=50 proto=icmp chksum=0x7e7c src=104.18.29.86 dst=172.24.188.141 |<ICMP  type=echo-reply code=0 chksum=0xee45 id=0x0 seq=0x0 |<Raw  load='XXXXXXXXXXX' |>>>
>>> p.show()
###[ IP ]###
  version= 4
  ihl= 5
  tos= 0x0
  len= 39
  id= 7244
  flags=
  frag= 0
  ttl= 50
  proto= icmp
  chksum= 0x7e7c
  src= 104.18.29.86
  dst= 172.24.188.141
  \options\
###[ ICMP ]###
     type= echo-reply
     code= 0
     chksum= 0xee45
     id= 0x0
     seq= 0x0
###[ Raw ]###
        load= 'XXXXXXXXXXX'

>>> sr1(IP(dst="192.168.5.1")/UDP()/DNS(rd=1,qd=DNSQR(qname="www.s...: lashdot.org")))
Begin emission:
Finished sending 1 packets.

^C
Received 0 packets, got 0 answers, remaining 1 packets
>>> sr(IP(dst="192.168.8.1")/TCP(dport=[21,22,23]))
Begin emission:
.Finished sending 3 packets.
*
.^C
Received 3 packets, got 1 answers, remaining 2 packets
(<Results: TCP:1 UDP:0 ICMP:0 Other:0>,
 <Unanswered: TCP:2 UDP:0 ICMP:0 Other:0>)
```
