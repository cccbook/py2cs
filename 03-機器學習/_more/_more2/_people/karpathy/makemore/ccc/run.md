(minGPT) PS D:\ccc\code\py\karpathy\makemore> python makemore.py -i names.txt -o names
{'input_file': 'names.txt', 'work_dir': 'names', 'resume': False, 'sample_only': False, 'num_workers': 4, 'max_steps': -1, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'transformer', 'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'n_embd2': 64, 'batch_size': 32, 'learning_rate': 0.0005, 'weight_decay': 0.01}
number of examples in the dataset: 32033
max word length: 15
number of unique characters in the vocabulary: 26
vocabulary:
abcdefghijklmnopqrstuvwxyz
split up the dataset into 31033 training examples and 1000 test examples
dataset determined that: vocab_size=27, block_size=16
number of parameters: 0.20M
model #params: 204544
step 0 | loss 3.5123 | step time 4294.28ms
step 10 | loss 2.8783 | step time 933.31ms
step 20 | loss 2.7156 | step time 991.80ms
step 30 | loss 2.5582 | step time 885.41ms
step 40 | loss 2.4769 | step time 994.40ms
step 50 | loss 2.5487 | step time 881.11ms
step 60 | loss 2.4465 | step time 953.52ms
step 70 | loss 2.4591 | step time 1087.00ms
step 80 | loss 2.3582 | step time 960.42ms
step 90 | loss 2.3096 | step time 927.09ms
step 100 | loss 2.5206 | step time 861.05ms
step 110 | loss 2.3936 | step time 906.35ms
step 120 | loss 2.3326 | step time 879.29ms
step 130 | loss 2.2815 | step time 854.34ms
step 140 | loss 2.4345 | step time 974.90ms
step 150 | loss 2.2379 | step time 881.46ms
step 160 | loss 2.1985 | step time 962.76ms
step 170 | loss 2.2849 | step time 1488.09ms
step 180 | loss 2.1690 | step time 983.97ms
step 190 | loss 2.2933 | step time 892.47ms
step 200 | loss 2.4520 | step time 886.18ms
--------------------------------------------------------------------------------
1 samples that are in train:
rynn
0 samples that are in test:
9 samples that are new:
a

stelas
brymyn
laradike
gharynqo
sameyn
jerily
yayzaah
--------------------------------------------------------------------------------
step 210 | loss 2.3004 | step time 860.30ms
step 220 | loss 2.2905 | step time 862.69ms
step 230 | loss 2.2604 | step time 1306.60ms
step 240 | loss 2.4505 | step time 1179.59ms
step 250 | loss 2.2336 | step time 1311.16ms
step 260 | loss 2.2900 | step time 979.79ms
step 270 | loss 2.2632 | step time 931.05ms
step 280 | loss 2.2970 | step time 929.15ms
step 290 | loss 2.2322 | step time 977.26ms
step 300 | loss 2.2271 | step time 1144.01ms
step 310 | loss 2.2325 | step time 859.37ms
step 320 | loss 2.3342 | step time 1127.37ms
step 330 | loss 2.2393 | step time 1202.77ms
step 340 | loss 2.1776 | step time 1038.42ms
step 350 | loss 2.3112 | step time 1034.24ms
step 360 | loss 2.1904 | step time 1062.41ms
step 370 | loss 2.2111 | step time 859.34ms
step 380 | loss 2.2426 | step time 981.43ms
step 390 | loss 2.1987 | step time 947.09ms
step 400 | loss 2.1702 | step time 859.23ms
--------------------------------------------------------------------------------
1 samples that are in train:
kennan
0 samples that are in test:
9 samples that are new:
zaredian
fenanne
rozri
bulian
rok
daita
eyn
estmtonn
redyzz
--------------------------------------------------------------------------------
step 410 | loss 2.1779 | step time 938.15ms
step 420 | loss 2.2607 | step time 1444.44ms
step 430 | loss 2.2063 | step time 935.41ms
step 440 | loss 2.2268 | step time 1369.44ms
step 450 | loss 2.2926 | step time 865.45ms
step 460 | loss 2.2584 | step time 890.62ms
step 470 | loss 2.1975 | step time 859.39ms
step 480 | loss 2.1548 | step time 875.00ms
step 490 | loss 2.2487 | step time 865.85ms
step 500 | loss 2.2251 | step time 874.99ms
step 500 train loss: 2.2131574153900146 test loss: 2.1970791816711426
test loss 2.1970791816711426 is the best so far, saving model to names\model.pt
step 510 | loss 2.2429 | step time 1557.97ms
step 520 | loss 2.1078 | step time 1163.30ms
step 530 | loss 2.3047 | step time 865.88ms
step 540 | loss 2.2194 | step time 863.75ms
step 550 | loss 2.1561 | step time 864.03ms
step 560 | loss 2.1503 | step time 861.14ms
step 570 | loss 2.2725 | step time 874.77ms
step 580 | loss 2.3158 | step time 926.97ms
step 590 | loss 2.1851 | step time 874.03ms
step 600 | loss 2.1109 | step time 907.66ms
--------------------------------------------------------------------------------
0 samples that are in train:
1 samples that are in test:
karen
9 samples that are new:
mareltad
chony
maiqyn
elmee
emaria
brizer
derah
alirius
brani
--------------------------------------------------------------------------------
step 610 | loss 2.1956 | step time 859.32ms
step 620 | loss 2.1942 | step time 918.62ms
step 630 | loss 2.2378 | step time 883.43ms
step 640 | loss 2.2287 | step time 1000.35ms
step 650 | loss 2.1940 | step time 1060.58ms
step 660 | loss 2.1910 | step time 856.77ms
step 670 | loss 2.1560 | step time 1352.99ms
step 680 | loss 2.1495 | step time 831.73ms
step 690 | loss 2.1672 | step time 849.60ms
step 700 | loss 2.1439 | step time 843.74ms
step 710 | loss 2.1130 | step time 927.00ms
step 720 | loss 2.1620 | step time 1055.31ms
step 730 | loss 2.1677 | step time 1266.76ms
step 740 | loss 2.0700 | step time 897.67ms
step 750 | loss 2.2258 | step time 1937.43ms
step 760 | loss 2.1848 | step time 916.42ms
step 770 | loss 2.2298 | step time 865.46ms
step 780 | loss 2.1519 | step time 1086.86ms
step 790 | loss 2.1820 | step time 1043.01ms
step 800 | loss 2.2200 | step time 997.98ms
--------------------------------------------------------------------------------
1 samples that are in train:
jakiyah
0 samples that are in test:
9 samples that are new:
arrema
anbista
bravin
laevincka
naqsin
ebrie
zejulon
roosiy
uesmir
--------------------------------------------------------------------------------
step 810 | loss 2.0950 | step time 858.36ms
step 820 | loss 2.1674 | step time 1038.13ms
step 830 | loss 2.1231 | step time 962.26ms
step 840 | loss 2.2643 | step time 964.61ms
step 850 | loss 2.1452 | step time 906.23ms
step 860 | loss 2.1040 | step time 1237.99ms
step 870 | loss 2.1471 | step time 1053.00ms
step 880 | loss 2.1540 | step time 1028.99ms
step 890 | loss 2.0456 | step time 1747.24ms
step 900 | loss 2.1628 | step time 849.81ms
step 910 | loss 2.2084 | step time 1533.21ms
step 920 | loss 2.2160 | step time 1145.06ms
step 930 | loss 2.2057 | step time 2956.62ms
step 940 | loss 2.0253 | step time 1020.00ms
step 950 | loss 2.1269 | step time 1065.25ms
step 960 | loss 2.2508 | step time 988.30ms
step 970 | loss 2.2674 | step time 1095.87ms
step 980 | loss 2.2671 | step time 848.92ms
step 990 | loss 2.1900 | step time 906.36ms
step 1000 | loss 2.2193 | step time 1177.00ms
step 1000 train loss: 2.144432783126831 test loss: 2.1338284015655518
test loss 2.1338284015655518 is the best so far, saving model to names\model.pt
--------------------------------------------------------------------------------
1 samples that are in train:
ariona
0 samples that are in test:
9 samples that are new:
killea
lomira
noren
braqan
brsen
daskima
koon
michayn
carge
--------------------------------------------------------------------------------
step 1010 | loss 2.1595 | step time 843.78ms
step 1020 | loss 2.2673 | step time 1076.03ms
step 1030 | loss 2.1428 | step time 1215.01ms
step 1040 | loss 2.0295 | step time 946.75ms
step 1050 | loss 2.2474 | step time 893.25ms
step 1060 | loss 2.1872 | step time 859.36ms
step 1070 | loss 2.1559 | step time 914.85ms
step 1080 | loss 2.0333 | step time 1047.72ms
step 1090 | loss 2.0651 | step time 979.35ms
step 1100 | loss 2.2640 | step time 843.78ms
step 1110 | loss 2.1855 | step time 1004.76ms
step 1120 | loss 2.0616 | step time 1058.99ms
step 1130 | loss 2.1546 | step time 1357.98ms
step 1140 | loss 2.2129 | step time 941.08ms
step 1150 | loss 2.1246 | step time 843.73ms
step 1160 | loss 2.0678 | step time 860.17ms
step 1170 | loss 2.1958 | step time 843.67ms
step 1180 | loss 2.2085 | step time 915.06ms
step 1190 | loss 2.1656 | step time 1069.32ms
step 1200 | loss 2.1061 | step time 1027.62ms
--------------------------------------------------------------------------------
2 samples that are in train:
markon
alea
0 samples that are in test:
8 samples that are new:
basny
kariyanl
zacri
rackces
yiom
hanela
halini
belleaja
--------------------------------------------------------------------------------
step 1210 | loss 2.1327 | step time 936.65ms
step 1220 | loss 2.1584 | step time 865.36ms
step 1230 | loss 2.2969 | step time 897.95ms
step 1240 | loss 2.2102 | step time 881.80ms
step 1250 | loss 1.9938 | step time 891.61ms
step 1260 | loss 2.1887 | step time 853.09ms
step 1270 | loss 2.0938 | step time 843.82ms
step 1280 | loss 2.1825 | step time 964.71ms
step 1290 | loss 2.0409 | step time 1155.08ms
step 1300 | loss 2.2894 | step time 1398.12ms
step 1310 | loss 2.0775 | step time 1020.87ms
step 1320 | loss 2.1478 | step time 988.70ms
step 1330 | loss 2.1893 | step time 1318.56ms
step 1340 | loss 2.1094 | step time 956.59ms
step 1350 | loss 2.0656 | step time 887.46ms
step 1360 | loss 2.2365 | step time 2201.31ms
step 1370 | loss 2.0507 | step time 1373.85ms
step 1380 | loss 2.1900 | step time 1192.42ms
step 1390 | loss 2.1151 | step time 1070.42ms
step 1400 | loss 2.3085 | step time 1092.89ms
--------------------------------------------------------------------------------
3 samples that are in train:
aira
reyan
gianno
0 samples that are in test:
7 samples that are new:
gan
zerna
zeyra
nakaysen
intoney
fanelie
jakena
--------------------------------------------------------------------------------
step 1410 | loss 2.1179 | step time 912.91ms
step 1420 | loss 2.0901 | step time 937.21ms
step 1430 | loss 2.1107 | step time 859.37ms
step 1440 | loss 2.2889 | step time 1927.36ms
step 1450 | loss 2.1344 | step time 959.20ms
step 1460 | loss 2.1540 | step time 1140.33ms
step 1470 | loss 2.0606 | step time 1034.77ms
step 1480 | loss 2.1514 | step time 916.90ms
step 1490 | loss 1.9697 | step time 843.68ms
step 1500 | loss 2.2233 | step time 914.28ms
step 1500 train loss: 2.1263296604156494 test loss: 2.106513023376465
test loss 2.106513023376465 is the best so far, saving model to names\model.pt
step 1510 | loss 2.0080 | step time 892.40ms
step 1520 | loss 2.1770 | step time 843.76ms
step 1530 | loss 2.0222 | step time 843.85ms
step 1540 | loss 2.0511 | step time 859.67ms
step 1550 | loss 2.1075 | step time 859.39ms
step 1560 | loss 2.1237 | step time 843.74ms
step 1570 | loss 2.3056 | step time 848.89ms
step 1580 | loss 2.1399 | step time 865.58ms
step 1590 | loss 2.1000 | step time 854.21ms
step 1600 | loss 2.2152 | step time 1255.35ms
--------------------------------------------------------------------------------
1 samples that are in train:
lala
1 samples that are in test:
navina
8 samples that are new:
arceline
braola
shaleigh
kastie
eyhiahia
rix
aravina
zinjus
--------------------------------------------------------------------------------
step 1610 | loss 2.0751 | step time 870.71ms
step 1620 | loss 2.0437 | step time 1051.46ms
step 1630 | loss 2.1870 | step time 1133.23ms
step 1640 | loss 2.1368 | step time 851.56ms
step 1650 | loss 2.0823 | step time 847.54ms
step 1660 | loss 2.0318 | step time 885.76ms
step 1670 | loss 2.0540 | step time 882.61ms
step 1680 | loss 2.1747 | step time 994.75ms
step 1690 | loss 2.1384 | step time 987.52ms
step 1700 | loss 2.1337 | step time 876.07ms
step 1710 | loss 2.1202 | step time 862.03ms
step 1720 | loss 2.1522 | step time 859.34ms
step 1730 | loss 2.0467 | step time 1070.17ms
step 1740 | loss 2.1764 | step time 1697.02ms
step 1750 | loss 2.1723 | step time 1413.99ms
step 1760 | loss 2.0704 | step time 846.04ms
step 1770 | loss 2.1440 | step time 1107.99ms
step 1780 | loss 2.1239 | step time 932.20ms
step 1790 | loss 1.9858 | step time 898.08ms
step 1800 | loss 2.1196 | step time 963.77ms
--------------------------------------------------------------------------------
1 samples that are in train:
amaryah
0 samples that are in test:
9 samples that are new:
yeyole
avayaann
yenncyy
zhyver
shanja
calaapkerlc
kenshie
soryn
jeshin
--------------------------------------------------------------------------------
step 1810 | loss 2.1631 | step time 864.38ms
step 1820 | loss 2.0679 | step time 1157.70ms
step 1830 | loss 2.0949 | step time 956.29ms
step 1840 | loss 2.0490 | step time 875.81ms
step 1850 | loss 2.0513 | step time 859.39ms
step 1860 | loss 2.1086 | step time 892.96ms
step 1870 | loss 2.0346 | step time 1057.44ms
step 1880 | loss 2.0911 | step time 828.14ms
step 1890 | loss 1.9909 | step time 1237.31ms
step 1900 | loss 2.0654 | step time 1365.01ms
step 1910 | loss 2.1291 | step time 873.00ms
step 1920 | loss 2.0480 | step time 867.00ms
step 1930 | loss 2.0651 | step time 849.00ms
step 1940 | loss 2.1276 | step time 839.00ms
step 1950 | loss 2.1058 | step time 852.00ms
step 1960 | loss 2.1387 | step time 848.00ms
step 1970 | loss 2.0311 | step time 855.99ms
step 1980 | loss 2.1762 | step time 849.00ms
step 1990 | loss 2.1460 | step time 849.00ms
step 2000 | loss 2.0737 | step time 839.00ms
step 2000 train loss: 2.085818290710449 test loss: 2.0845537185668945
test loss 2.0845537185668945 is the best so far, saving model to names\model.pt
--------------------------------------------------------------------------------
2 samples that are in train:
daylen
jaziyah
0 samples that are in test:
8 samples that are new:
achith
selyn
trewsor
kherlian
pestoy
mancersera
rovyna
khiste
--------------------------------------------------------------------------------
step 2010 | loss 2.0322 | step time 861.00ms
step 2020 | loss 2.0681 | step time 843.00ms
step 2030 | loss 2.1537 | step time 845.00ms
step 2040 | loss 2.2794 | step time 861.00ms
step 2050 | loss 1.9946 | step time 863.02ms
step 2060 | loss 2.0836 | step time 853.00ms
step 2070 | loss 2.0055 | step time 840.00ms
step 2080 | loss 2.1480 | step time 851.00ms
step 2090 | loss 2.0520 | step time 847.00ms
step 2100 | loss 2.0367 | step time 846.00ms
step 2110 | loss 2.0109 | step time 857.00ms
step 2120 | loss 1.9517 | step time 846.00ms
step 2130 | loss 2.1458 | step time 850.00ms
step 2140 | loss 2.0210 | step time 1093.00ms
step 2150 | loss 1.9763 | step time 857.99ms
step 2160 | loss 2.0512 | step time 875.01ms
step 2170 | loss 2.0989 | step time 868.00ms
step 2180 | loss 2.0220 | step time 839.00ms
step 2190 | loss 2.0249 | step time 839.01ms
step 2200 | loss 2.0524 | step time 886.00ms
--------------------------------------------------------------------------------
1 samples that are in train:
kahri
1 samples that are in test:
ariane
8 samples that are new:
izer
eviviah
mamari
musriodiat
dakon
lanyile
amaerah
kaidy
--------------------------------------------------------------------------------
step 2210 | loss 1.9425 | step time 863.00ms
step 2220 | loss 2.1292 | step time 870.01ms
step 2230 | loss 2.1767 | step time 847.00ms
step 2240 | loss 2.1400 | step time 864.02ms
step 2250 | loss 2.0004 | step time 855.00ms
step 2260 | loss 1.9792 | step time 834.00ms
step 2270 | loss 2.0254 | step time 866.00ms
step 2280 | loss 2.0927 | step time 859.00ms
step 2290 | loss 2.0277 | step time 863.00ms
step 2300 | loss 2.2087 | step time 851.00ms
step 2310 | loss 1.9600 | step time 840.00ms
step 2320 | loss 2.1122 | step time 850.00ms
step 2330 | loss 1.9909 | step time 846.00ms
step 2340 | loss 2.0229 | step time 913.00ms
step 2350 | loss 2.0648 | step time 855.00ms
step 2360 | loss 2.1707 | step time 841.99ms
step 2370 | loss 2.1374 | step time 837.00ms
step 2380 | loss 2.1873 | step time 842.00ms
step 2390 | loss 2.0143 | step time 867.00ms
step 2400 | loss 2.2086 | step time 843.00ms
--------------------------------------------------------------------------------
1 samples that are in train:
kensleigh
0 samples that are in test:
9 samples that are new:
avala
vanciya
sanasaila
sibaly
hinzis
horcer
aawey
ceritt
moytn
--------------------------------------------------------------------------------
step 2410 | loss 2.0819 | step time 852.00ms
step 2420 | loss 2.1072 | step time 864.99ms
step 2430 | loss 2.1547 | step time 850.00ms
step 2440 | loss 2.0833 | step time 837.00ms
step 2450 | loss 2.2108 | step time 859.00ms
step 2460 | loss 2.0134 | step time 852.00ms
step 2470 | loss 2.0652 | step time 854.00ms
step 2480 | loss 1.9818 | step time 844.00ms
step 2490 | loss 2.0391 | step time 844.00ms
step 2500 | loss 2.0909 | step time 849.00ms
step 2500 train loss: 2.038792133331299 test loss: 2.0630156993865967
test loss 2.0630156993865967 is the best so far, saving model to names\model.pt
step 2510 | loss 1.9410 | step time 847.00ms
step 2520 | loss 1.8769 | step time 855.00ms
step 2530 | loss 2.0236 | step time 860.00ms
step 2540 | loss 2.1405 | step time 851.00ms
step 2550 | loss 2.0907 | step time 931.00ms
step 2560 | loss 1.9995 | step time 828.33ms
step 2570 | loss 2.1775 | step time 828.21ms
step 2580 | loss 1.9577 | step time 843.90ms
step 2590 | loss 2.1396 | step time 828.12ms
step 2600 | loss 1.9581 | step time 812.48ms
--------------------------------------------------------------------------------
1 samples that are in train:
jacky
0 samples that are in test:
9 samples that are new:
emraod
daita
ziron
ayanie
jamariana
makalene
sodepho
roley
caraa
--------------------------------------------------------------------------------
step 2610 | loss 2.0179 | step time 843.70ms
step 2620 | loss 2.2431 | step time 833.13ms
step 2630 | loss 2.0115 | step time 828.23ms
step 2640 | loss 2.0210 | step time 843.76ms
step 2650 | loss 1.9600 | step time 834.25ms
step 2660 | loss 2.0640 | step time 843.73ms
step 2670 | loss 2.0817 | step time 843.75ms
step 2680 | loss 2.0765 | step time 833.57ms
step 2690 | loss 2.0050 | step time 859.70ms
step 2700 | loss 2.0524 | step time 828.39ms
step 2710 | loss 2.1439 | step time 848.82ms
step 2720 | loss 2.0626 | step time 828.06ms
step 2730 | loss 2.0509 | step time 828.15ms
step 2740 | loss 2.1548 | step time 834.03ms
step 2750 | loss 1.9833 | step time 843.76ms
step 2760 | loss 2.0513 | step time 843.70ms
step 2770 | loss 2.0711 | step time 818.81ms
step 2780 | loss 2.0188 | step time 843.80ms
step 2790 | loss 2.0626 | step time 812.54ms
step 2800 | loss 1.9842 | step time 817.37ms
--------------------------------------------------------------------------------
0 samples that are in train:
0 samples that are in test:
10 samples that are new:
nazira
nayas
leetondre
alexand
hazzlee
lenia
omerion
yuitke
sitthas
baygon
--------------------------------------------------------------------------------
step 2810 | loss 2.0872 | step time 812.50ms
step 2820 | loss 1.9312 | step time 843.74ms
step 2830 | loss 2.0978 | step time 828.02ms
step 2840 | loss 1.9950 | step time 828.14ms
step 2850 | loss 2.0104 | step time 828.15ms
step 2860 | loss 2.0438 | step time 828.39ms
step 2870 | loss 2.1093 | step time 859.52ms
step 2880 | loss 2.0243 | step time 843.80ms
step 2890 | loss 2.0012 | step time 843.85ms
step 2900 | loss 2.0326 | step time 833.02ms
step 2910 | loss 1.9765 | step time 828.09ms
step 2920 | loss 2.0888 | step time 828.13ms
step 2930 | loss 2.1309 | step time 832.78ms
step 2940 | loss 2.1171 | step time 906.25ms
step 2950 | loss 2.0525 | step time 921.89ms
step 2960 | loss 2.0015 | step time 812.50ms
step 2970 | loss 2.1111 | step time 968.76ms
step 2980 | loss 1.8874 | step time 1099.18ms
step 2990 | loss 2.0058 | step time 865.53ms
step 3000 | loss 2.1363 | step time 872.99ms
step 3000 train loss: 2.0318901538848877 test loss: 2.0527865886688232
test loss 2.0527865886688232 is the best so far, saving model to names\model.pt
--------------------------------------------------------------------------------
1 samples that are in train:
kedrick
0 samples that are in test:
9 samples that are new:
nazliya
fobathon
allin
vallaw
zyfan
jakhyra
kassir
kayner
trieston
--------------------------------------------------------------------------------
step 3010 | loss 2.1113 | step time 1063.70ms
step 3020 | loss 1.9803 | step time 1072.21ms
step 3030 | loss 2.0899 | step time 1445.00ms
step 3040 | loss 2.1327 | step time 1301.54ms
step 3050 | loss 1.8516 | step time 836.37ms
step 3060 | loss 1.9586 | step time 845.44ms
step 3070 | loss 1.9413 | step time 957.92ms
step 3080 | loss 2.0577 | step time 970.77ms
step 3090 | loss 2.0224 | step time 867.55ms
step 3100 | loss 2.0058 | step time 872.47ms
step 3110 | loss 1.9506 | step time 1020.99ms
step 3120 | loss 1.9697 | step time 856.84ms
step 3130 | loss 2.1286 | step time 1196.99ms
step 3140 | loss 2.1525 | step time 843.70ms
step 3150 | loss 2.1245 | step time 950.22ms
step 3160 | loss 2.0434 | step time 828.14ms
step 3170 | loss 1.9806 | step time 1180.07ms
step 3180 | loss 2.1425 | step time 851.12ms
step 3190 | loss 1.9003 | step time 871.25ms
step 3200 | loss 2.0925 | step time 897.24ms
--------------------------------------------------------------------------------
4 samples that are in train:
avrielle
anani
amira
roza
0 samples that are in test:
6 samples that are new:
aldura
sru
zakad
maxsion
vaneela
neora
--------------------------------------------------------------------------------
step 3210 | loss 1.9848 | step time 903.14ms
step 3220 | loss 2.0095 | step time 1109.87ms
step 3230 | loss 2.1086 | step time 1047.30ms
step 3240 | loss 1.9723 | step time 926.24ms
step 3250 | loss 2.1043 | step time 951.89ms
step 3260 | loss 1.9312 | step time 1212.69ms
step 3270 | loss 2.0503 | step time 1072.44ms
step 3280 | loss 2.1587 | step time 880.60ms
step 3290 | loss 1.9783 | step time 951.12ms
step 3300 | loss 1.9355 | step time 921.87ms
step 3310 | loss 1.8515 | step time 905.61ms
step 3320 | loss 2.0442 | step time 1195.14ms
step 3330 | loss 2.0439 | step time 844.04ms
step 3340 | loss 2.1258 | step time 900.69ms
step 3350 | loss 2.0023 | step time 1001.05ms
step 3360 | loss 1.9164 | step time 1136.38ms
step 3370 | loss 1.9907 | step time 898.24ms
step 3380 | loss 1.9416 | step time 897.37ms
step 3390 | loss 2.1038 | step time 866.69ms
step 3400 | loss 1.9887 | step time 882.61ms
--------------------------------------------------------------------------------
5 samples that are in train:
kyere
jenay
brayon
linna
brayan
0 samples that are in test:
5 samples that are new:
toriu
dachi
hugesi
famero
zyer
--------------------------------------------------------------------------------
step 3410 | loss 2.1648 | step time 940.73ms
step 3420 | loss 1.9292 | step time 837.58ms
step 3430 | loss 2.0721 | step time 905.42ms
step 3440 | loss 1.9705 | step time 888.12ms
step 3450 | loss 1.9753 | step time 953.23ms
step 3460 | loss 2.0166 | step time 911.72ms
step 3470 | loss 2.0240 | step time 911.94ms
step 3480 | loss 1.9421 | step time 826.00ms
step 3490 | loss 1.9578 | step time 961.12ms
step 3500 | loss 2.0753 | step time 861.44ms
step 3500 train loss: 2.0194191932678223 test loss: 2.030223846435547
test loss 2.030223846435547 is the best so far, saving model to names\model.pt
step 3510 | loss 1.9909 | step time 996.32ms
step 3520 | loss 2.0956 | step time 929.60ms
step 3530 | loss 1.9952 | step time 1240.74ms
step 3540 | loss 1.9862 | step time 878.70ms
step 3550 | loss 1.9655 | step time 913.88ms
step 3560 | loss 1.9979 | step time 2247.44ms
step 3570 | loss 2.0396 | step time 862.54ms
step 3580 | loss 2.0686 | step time 908.81ms
step 3590 | loss 1.9473 | step time 895.24ms
step 3600 | loss 2.0043 | step time 940.89ms
--------------------------------------------------------------------------------
2 samples that are in train:
damier
daniele
0 samples that are in test:
8 samples that are new:
ayolla
dexon
dryston
shakida
rega
azioh
tevricko
pegber
--------------------------------------------------------------------------------
step 3610 | loss 2.0541 | step time 882.30ms
step 3620 | loss 1.9951 | step time 843.64ms
step 3630 | loss 1.9670 | step time 844.71ms
step 3640 | loss 1.9165 | step time 1156.00ms
step 3650 | loss 1.9913 | step time 1269.01ms
step 3660 | loss 1.9432 | step time 1118.00ms
step 3670 | loss 2.1755 | step time 1406.14ms
step 3680 | loss 1.9946 | step time 2678.02ms
step 3690 | loss 1.9457 | step time 1261.99ms
step 3700 | loss 2.0749 | step time 1265.58ms
step 3710 | loss 1.9712 | step time 985.43ms
step 3720 | loss 1.9718 | step time 1083.99ms
step 3730 | loss 1.9133 | step time 1031.07ms
step 3740 | loss 2.1228 | step time 1101.51ms
step 3750 | loss 1.9698 | step time 1081.00ms
step 3760 | loss 1.9648 | step time 892.00ms
step 3770 | loss 2.0102 | step time 893.00ms
step 3780 | loss 2.0672 | step time 851.43ms
step 3790 | loss 1.9823 | step time 952.00ms
step 3800 | loss 1.9637 | step time 954.00ms
--------------------------------------------------------------------------------
1 samples that are in train:
layne
0 samples that are in test:
9 samples that are new:
bley
jaoven
gracere
brica
haylon
zayvia
brustalop
swamby
nily
--------------------------------------------------------------------------------
step 3810 | loss 2.0036 | step time 904.02ms
step 3820 | loss 1.8960 | step time 1009.36ms
step 3830 | loss 1.9767 | step time 909.00ms
step 3840 | loss 1.9741 | step time 930.37ms
step 3850 | loss 2.0002 | step time 901.67ms
step 3860 | loss 1.8659 | step time 1544.00ms
step 3870 | loss 2.0750 | step time 1451.05ms
step 3880 | loss 2.0181 | step time 2319.00ms
step 3890 | loss 1.9732 | step time 934.99ms
step 3900 | loss 2.0905 | step time 948.00ms
step 3910 | loss 2.0724 | step time 965.58ms
step 3920 | loss 2.0535 | step time 1645.13ms
step 3930 | loss 2.0142 | step time 1151.01ms
step 3940 | loss 2.0427 | step time 1587.91ms
step 3950 | loss 1.9188 | step time 1182.55ms
step 3960 | loss 2.0275 | step time 942.00ms
step 3970 | loss 2.0635 | step time 1427.57ms
step 3980 | loss 2.0201 | step time 974.00ms
step 3990 | loss 2.0450 | step time 1749.26ms
step 4000 | loss 2.0084 | step time 2867.12ms
step 4000 train loss: 2.00943660736084 test loss: 2.0277249813079834
test loss 2.0277249813079834 is the best so far, saving model to names\model.pt
--------------------------------------------------------------------------------
2 samples that are in train:
nevia
amelia
0 samples that are in test:
8 samples that are new:
yazar
semangey
oretti
ju
nyeor
jayelly
zinar
aabiley
--------------------------------------------------------------------------------
step 4010 | loss 2.1573 | step time 896.99ms
step 4020 | loss 2.0093 | step time 1632.46ms
step 4030 | loss 2.0240 | step time 1113.58ms
step 4040 | loss 1.9907 | step time 1261.99ms
step 4050 | loss 1.9969 | step time 1109.00ms
step 4060 | loss 1.8742 | step time 867.28ms
step 4070 | loss 2.0293 | step time 1109.00ms
step 4080 | loss 1.9741 | step time 1148.33ms
step 4090 | loss 2.1274 | step time 2001.72ms
step 4100 | loss 1.9584 | step time 1106.70ms
step 4110 | loss 1.9983 | step time 1086.09ms
step 4120 | loss 2.0785 | step time 1075.99ms
step 4130 | loss 2.0182 | step time 833.90ms
step 4140 | loss 1.9154 | step time 954.64ms
step 4150 | loss 2.0095 | step time 1193.11ms
step 4160 | loss 2.1650 | step time 897.55ms
step 4170 | loss 2.0284 | step time 1707.06ms
step 4180 | loss 1.9645 | step time 1280.57ms
step 4190 | loss 1.9311 | step time 828.10ms
step 4200 | loss 2.2455 | step time 1017.00ms
--------------------------------------------------------------------------------
1 samples that are in train:
ahmara
0 samples that are in test:
9 samples that are new:
dadi
noeeber
masilynn
tacuman
marisha
asi
tambrion
kaninl
isaor
--------------------------------------------------------------------------------
step 4210 | loss 2.0351 | step time 935.88ms
step 4220 | loss 1.9600 | step time 1178.41ms
step 4230 | loss 1.9046 | step time 964.60ms
step 4240 | loss 2.0201 | step time 1103.01ms
step 4250 | loss 1.8909 | step time 1055.72ms
step 4260 | loss 2.0207 | step time 971.52ms
step 4270 | loss 2.0837 | step time 1034.27ms
step 4280 | loss 1.8924 | step time 1276.01ms
step 4290 | loss 1.8789 | step time 1092.92ms
step 4300 | loss 1.9697 | step time 1224.38ms
step 4310 | loss 2.0198 | step time 978.09ms
step 4320 | loss 1.9306 | step time 1138.26ms
step 4330 | loss 2.0388 | step time 1300.01ms
step 4340 | loss 1.9308 | step time 1236.99ms
step 4350 | loss 2.0118 | step time 1083.99ms
step 4360 | loss 2.1199 | step time 1036.69ms
step 4370 | loss 1.9258 | step time 1404.38ms
step 4380 | loss 2.0108 | step time 1044.70ms
step 4390 | loss 2.0027 | step time 855.08ms
step 4400 | loss 2.0922 | step time 1282.19ms
--------------------------------------------------------------------------------
4 samples that are in train:
aleiah
mahad
joe
daeson
0 samples that are in test:
6 samples that are new:
marduben
everma
berky
pinae
alihel
naelmi
--------------------------------------------------------------------------------
step 4410 | loss 1.9980 | step time 949.97ms
step 4420 | loss 2.0443 | step time 1062.01ms
step 4430 | loss 2.1127 | step time 918.91ms
step 4440 | loss 1.9441 | step time 855.63ms
step 4450 | loss 2.0073 | step time 892.65ms
step 4460 | loss 1.9664 | step time 923.01ms
step 4470 | loss 1.9612 | step time 1293.00ms
step 4480 | loss 1.9813 | step time 2159.00ms
step 4490 | loss 2.0862 | step time 845.40ms
step 4500 | loss 1.9902 | step time 2232.06ms
step 4500 train loss: 1.9769797325134277 test loss: 2.0127508640289307
test loss 2.0127508640289307 is the best so far, saving model to names\model.pt
step 4510 | loss 2.0959 | step time 1724.91ms
step 4520 | loss 2.0979 | step time 1016.35ms
step 4530 | loss 1.8699 | step time 1364.43ms
step 4540 | loss 2.0754 | step time 875.48ms
step 4550 | loss 1.9928 | step time 1260.58ms
step 4560 | loss 2.1024 | step time 860.00ms
step 4570 | loss 1.9300 | step time 934.99ms
step 4580 | loss 2.0459 | step time 887.00ms
step 4590 | loss 1.8034 | step time 888.03ms
step 4600 | loss 1.9775 | step time 872.99ms
--------------------------------------------------------------------------------
2 samples that are in train:
ryah
laurie
0 samples that are in test:
8 samples that are new:
emberres
kasar
faylin
marica
hittri
mahson
josidy
zeberl
--------------------------------------------------------------------------------
step 4610 | loss 1.9058 | step time 907.01ms
step 4620 | loss 1.9313 | step time 940.00ms
step 4630 | loss 2.0116 | step time 849.00ms
step 4640 | loss 1.9908 | step time 843.97ms
step 4650 | loss 2.0511 | step time 895.69ms
step 4660 | loss 2.0335 | step time 978.34ms
step 4670 | loss 1.9057 | step time 975.69ms
step 4680 | loss 2.0061 | step time 1174.99ms
step 4690 | loss 2.0292 | step time 1018.99ms
step 4700 | loss 1.8702 | step time 878.99ms
step 4710 | loss 2.1440 | step time 1056.99ms
step 4720 | loss 2.0605 | step time 1176.86ms
step 4730 | loss 2.0037 | step time 953.75ms
step 4740 | loss 1.8754 | step time 843.76ms
step 4750 | loss 1.9747 | step time 828.11ms
step 4760 | loss 2.0887 | step time 893.96ms
step 4770 | loss 2.1079 | step time 1200.00ms
step 4780 | loss 1.9687 | step time 826.38ms
step 4790 | loss 2.0237 | step time 939.03ms
step 4800 | loss 1.9907 | step time 1216.99ms
--------------------------------------------------------------------------------
3 samples that are in train:
lauren
kaylie
samah
0 samples that are in test:
7 samples that are new:
mage
beawd
mottell
saribett
nowafal
dae
auda
--------------------------------------------------------------------------------
step 4810 | loss 2.1100 | step time 875.60ms
step 4820 | loss 1.8445 | step time 894.68ms
step 4830 | loss 1.9779 | step time 843.47ms
step 4840 | loss 2.1631 | step time 828.12ms
step 4850 | loss 1.9536 | step time 1007.45ms
step 4860 | loss 1.9202 | step time 1063.54ms
step 4870 | loss 2.1302 | step time 1052.56ms
step 4880 | loss 2.0973 | step time 871.63ms
step 4890 | loss 2.0260 | step time 832.58ms
step 4900 | loss 2.0284 | step time 967.77ms
step 4910 | loss 1.9396 | step time 1741.01ms
step 4920 | loss 1.9854 | step time 1212.00ms
step 4930 | loss 2.0438 | step time 1788.94ms
step 4940 | loss 2.0181 | step time 859.68ms
step 4950 | loss 1.9992 | step time 947.80ms
step 4960 | loss 1.9599 | step time 858.34ms
step 4970 | loss 2.1311 | step time 843.76ms
step 4980 | loss 1.9683 | step time 824.27ms
step 4990 | loss 1.9949 | step time 843.80ms
step 5000 | loss 1.9307 | step time 1095.99ms
step 5000 train loss: 1.9757014513015747 test loss: 2.012585401535034
test loss 2.012585401535034 is the best so far, saving model to names\model.pt
--------------------------------------------------------------------------------
3 samples that are in train:
ady
brixton
kasir
0 samples that are in test:
7 samples that are new:
zyiana
tenslie
arba
calqualan
nassiya
praqudy
skylo
--------------------------------------------------------------------------------
step 5010 | loss 2.0247 | step time 1165.98ms
step 5020 | loss 2.0125 | step time 834.37ms
step 5030 | loss 1.7919 | step time 1104.00ms
step 5040 | loss 1.9408 | step time 1018.99ms
step 5050 | loss 1.8331 | step time 1067.44ms
step 5060 | loss 2.0509 | step time 1525.33ms
step 5070 | loss 1.9349 | step time 1021.68ms
step 5080 | loss 1.9793 | step time 867.84ms
step 5090 | loss 2.1417 | step time 846.04ms
step 5100 | loss 1.8277 | step time 1292.08ms
step 5110 | loss 1.9789 | step time 885.20ms
step 5120 | loss 1.9333 | step time 1614.99ms
step 5130 | loss 1.9486 | step time 967.63ms
step 5140 | loss 2.0128 | step time 893.45ms
step 5150 | loss 1.8981 | step time 992.01ms
step 5160 | loss 1.9703 | step time 865.88ms
step 5170 | loss 1.9574 | step time 848.98ms
step 5180 | loss 1.9711 | step time 1008.13ms
step 5190 | loss 1.9387 | step time 1209.81ms
step 5200 | loss 2.0278 | step time 832.72ms
--------------------------------------------------------------------------------
1 samples that are in train:
alira
0 samples that are in test:
9 samples that are new:
aniyash
aquyah
ppsianatinsse
cortson
zachynn
arries
raby
corwanci
carlle
--------------------------------------------------------------------------------
step 5210 | loss 2.0641 | step time 859.19ms
step 5220 | loss 1.9176 | step time 884.34ms
step 5230 | loss 1.9237 | step time 848.24ms
step 5240 | loss 2.0427 | step time 882.34ms
step 5250 | loss 1.9116 | step time 1038.66ms
step 5260 | loss 2.0134 | step time 864.06ms
step 5270 | loss 2.0046 | step time 977.41ms
step 5280 | loss 2.0615 | step time 828.13ms
step 5290 | loss 1.9161 | step time 828.08ms
step 5300 | loss 1.9847 | step time 897.85ms
step 5310 | loss 2.1173 | step time 906.64ms
step 5320 | loss 2.0648 | step time 1449.46ms
step 5330 | loss 1.9817 | step time 859.37ms
step 5340 | loss 1.9746 | step time 1015.82ms
step 5350 | loss 1.9293 | step time 843.72ms
step 5360 | loss 1.9322 | step time 877.93ms
step 5370 | loss 1.8589 | step time 828.12ms
step 5380 | loss 1.9436 | step time 869.12ms
step 5390 | loss 1.9266 | step time 869.90ms
step 5400 | loss 1.8913 | step time 1824.25ms
--------------------------------------------------------------------------------
2 samples that are in train:
kairan
brycen
0 samples that are in test:
8 samples that are new:
mosselala
awannki
kingger
chis
kubya
midela
denil
brisarde
--------------------------------------------------------------------------------
step 5410 | loss 1.9758 | step time 878.97ms
step 5420 | loss 1.9235 | step time 1049.73ms
step 5430 | loss 2.0526 | step time 843.17ms
step 5440 | loss 2.0663 | step time 1776.03ms
step 5450 | loss 1.8484 | step time 1128.00ms
step 5460 | loss 1.9555 | step time 1340.82ms
Traceback (most recent call last):
  File "D:\ccc\code\py\karpathy\makemore\makemore.py", line 684, in <module>
    loss.backward()
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\torch\_tensor.py", line 396, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\torch\autograd\__init__.py", line 173, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
KeyboardInterrupt
