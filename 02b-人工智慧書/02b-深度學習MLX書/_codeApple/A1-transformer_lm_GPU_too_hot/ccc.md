

在我的 iMac M3 上跑到 Iter 100 之後， 聽到 iMAC 內部風扇聲音很大，就不敢再跑了，怕燒掉 ...


```
(base) cccimac@cccimacdeiMac transformer_lm % python main.py --gpu
Training a transformer with 153.883 M parameters
Iter 10: Train loss 8.358, It/sec 0.258
Iter 20: Train loss 7.235, It/sec 0.271
Iter 30: Train loss 6.784, It/sec 0.282
Iter 40: Train loss 6.679, It/sec 0.282
Iter 50: Train loss 6.677, It/sec 0.247
Iter 60: Train loss 6.606, It/sec 0.336
Iter 70: Train loss 6.650, It/sec 0.364
Iter 80: Train loss 6.619, It/sec 0.386
Iter 90: Train loss 6.490, It/sec 0.418
Iter 100: Train loss 6.524, It/sec 0.388
Iter 110: Train loss 6.634, It/sec 0.325
Iter 120: Train loss 6.519, It/sec 0.505
^C^CTraceback (most recent call last):
  File "/Users/cccimac/Desktop/ccc/py2cs/02b-人工智慧書/02b-深度學習MLX書/_codeApple/mlx-examples/transformer_lm/main.py", line 220, in <module>
    main(args)
  File "/Users/cccimac/Desktop/ccc/py2cs/02b-人工智慧書/02b-深度學習MLX書/_codeApple/mlx-examples/transformer_lm/main.py", line 122, in main
    mx.eval(state)
KeyboardInterrupt
```