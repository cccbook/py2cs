# FPGA

* [Wikipedia:FPGA](https://en.wikipedia.org/wiki/Field-programmable_gate_array)
* [Wikipedia:Logic block](https://en.wikipedia.org/wiki/Logic_block)

> FPGA = #CLB + #Routing

## CLB

CLB = Look-up table (LUT) + FullAdder (FA) + Flip-Flop (FF)

![](https://upload.wikimedia.org/wikipedia/commons/1/1c/FPGA_cell_example.png)

圖中是 3 LUT, 但現在常用 5 LUT (應該是成本平衡考量)

## Routing

Switch box: 三種可能
1. 直通
2. 轉折
3. 斷線

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Switch_box.svg/1280px-Switch_box.svg.png)

## 參考

* https://zhuanlan.zhihu.com/p/340328985

基础的FPGA结构由以下几部分组成：

Look-up table(LUT): This element performs logic operations
Flip-Flop(FF): This register element stores the result of the LUT
Wires: These elements connect elements to one another
Input/Output(I/O) pads: These physical ports get data in and out of the FPGA

## LUT

* https://www.eet-china.com/mp/a106274.html

查找表（Look-Up-Table）简称为LUT，LUT本质上就是一个RAM。目前FPGA中多使用4输入的LUT，所以每一个LUT可以看成一个有4位地址线的 的RAM。当用户通过原理图或HDL语言描述了一个逻辑电路以后，PLD/FPGA开发软件会自动计算逻辑电路的所有可能结果，并把真值表（即结果）事先写入RAM，这样，每输入一个信号进行逻辑运算就等于输入一个地址进行查表，找出地址对应的内容，然后输出即可。



