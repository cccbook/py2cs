

## ccc: 能給出用 verilog, Verilator, Yosys, OpenRoad 進行 IC 設計的完整範例嗎？

ChatGPT:

以下是一個完整的範例，展示如何使用 Verilog、Verilator、Yosys 和 OpenROAD 進行簡單的 IC 設計流程。這裡的設計是以一個簡單的 4 位元加法器為例：

---

### 1. Verilog 設計

首先，我們定義一個簡單的 4 位元加法器模組：

```verilog
// adder.v
module adder (
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule
```

---

### 2. Verilator 模擬

用 Verilator 模擬設計，驗證功能是否正確。

#### 測試程式：
```cpp
// test_adder.cpp
#include <verilated.h>
#include "Vadder.h"

int main() {
    Vadder* adder = new Vadder;

    adder->a = 5;  // 0101
    adder->b = 3;  // 0011
    adder->eval();

    printf("Sum: %d\n", adder->sum);  // 應輸出 8

    delete adder;
    return 0;
}
```

#### 編譯與執行：
```bash
verilator --cc adder.v --exe test_adder.cpp
make -j -C obj_dir -f Vadder.mk Vadder
./obj_dir/Vadder
```

---

### 3. 使用 Yosys 合成 RTL

Yosys 將 Verilog 轉換為閘級網表。

#### Yosys 指令檔：
```tcl
# adder.ys
read_verilog adder.v
synth -top adder
write_verilog adder_synth.v
```

#### 執行 Yosys：
```bash
yosys -s adder.ys
```

---

### 4. 使用 OpenROAD 進行布局與布線

將合成後的網表進行布局布線，生成 GDSII。

#### OpenROAD 指令檔：
```tcl
# adder_floorplan.tcl
initialize_floorplan -die_area "0 0 1000 1000" -core_area "100 100 900 900"
read_verilog adder_synth.v
read_liberty my_lib.lib
read_lef my_tech.lef
place_io
global_placement
detailed_placement
global_routing
detailed_routing
write_gds adder.gds
```

#### 執行 OpenROAD：
```bash
openroad -exit adder_floorplan.tcl
```

---

### 所需文件

1. **標準單元庫檔案** (`my_lib.lib`, `my_tech.lef`)：
   - 這些檔案可從標準單元庫提供者獲取，例如 SkyWater 130nm 的開源庫。
   - 下載地址：[SkyWater PDK](https://github.com/google/skywater-pdk)

2. **GDSII 輸出檔案**：
   - 最終生成的 `adder.gds` 文件是設計的版圖。

---

### 工具安裝與設定

#### Verilator
```bash
sudo apt install verilator
```

#### Yosys
```bash
sudo apt install yosys
```

#### OpenROAD
[官方安裝指南](https://github.com/The-OpenROAD-Project/OpenROAD)

---

### 注意事項

1. **PDK 的準備**：  
   確保已安裝並配置 SkyWater PDK 或其他技術庫。

2. **測試與驗證**：  
   可以將模擬結果與布局布線後的結果進行比對，確保功能一致性。

這是一個從設計到實現的完整流程，適用於小型 IC 設計。如果有更複雜的需求，可以進一步整合時序分析（STA）和功耗分析工具，例如 OpenSTA。

