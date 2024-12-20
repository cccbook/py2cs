

用 pip3 install mlx-data 失敗的話，改用

```
$ git clone https://github.com/ml-explore/mlx-data.git
$ cd mlx-data
$ pip3 install .
```

然後執行

```
$ python main.py

...

Number of trainable params: 0.1493 M
Epoch    1 | Loss   15033.70 | Throughput  2362.14 im/s | Time     27.7 (s)
Epoch    2 | Loss   10544.95 | Throughput  2354.29 im/s | Time     25.8 (s)
Epoch    3 | Loss    9969.43 | Throughput  2356.20 im/s | Time     25.8 (s)
Epoch    4 | Loss    9670.92 | Throughput  2365.01 im/s | Time     25.7 (s)
Epoch    5 | Loss    9484.80 | Throughput  2353.03 im/s | Time     25.9 (s)
Epoch    6 | Loss    9342.81 | Throughput  2298.32 im/s | Time     26.7 (s)
Epoch    7 | Loss    9234.33 | Throughput  2266.42 im/s | Time     27.2 (s)
Epoch    8 | Loss    9147.80 | Throughput  2297.27 im/s | Time     26.8 (s)
Epoch    9 | Loss    9074.71 | Throughput  2306.99 im/s | Time     26.6 (s)
Epoch   10 | Loss    9009.72 | Throughput  2286.31 im/s | Time     26.9 (s)
Epoch   11 | Loss    8983.38 | Throughput  2300.50 im/s | Batch   280
```
