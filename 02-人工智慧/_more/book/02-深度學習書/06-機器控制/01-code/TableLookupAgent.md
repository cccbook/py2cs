我來實現一個基於查表法(Table-lookup)的智能體。查表法的核心思想是將狀態空間離散化，為每個離散狀態指定一個固定的動作。



```python
import numpy as np

class TableLookupAgent:
    def __init__(self, env):
        # 狀態空間離散化參數
        self.pos_bins = 3      # 位置分段數
        self.vel_bins = 3      # 速度分段數
        self.ang_bins = 6      # 角度分段數
        self.angvel_bins = 6   # 角速度分段數
        
        # 定義狀態空間的範圍
        self.pos_range = (-2.4, 2.4)
        self.vel_range = (-4.0, 4.0)
        self.ang_range = (-0.21, 0.21)
        self.angvel_range = (-4.0, 4.0)
        
        # 初始化動作表
        # 根據物理直覺設置預定的動作
        self.action_table = self._initialize_action_table()
        
    def reset(self, mode=None):
        return
        
    def _discretize_state(self, observation):
        """將連續狀態離散化"""
        pos, vel, ang, angvel = observation
        
        # 對每個狀態變量進行離散化
        pos_idx = self._discretize_value(pos, self.pos_range, self.pos_bins)
        vel_idx = self._discretize_value(vel, self.vel_range, self.vel_bins)
        ang_idx = self._discretize_value(ang, self.ang_range, self.ang_bins)
        angvel_idx = self._discretize_value(angvel, self.angvel_range, self.angvel_bins)
        
        return (pos_idx, vel_idx, ang_idx, angvel_idx)
        
    def _discretize_value(self, value, value_range, num_bins):
        """將單個值離散化到指定的區間"""
        min_val, max_val = value_range
        # 確保值在範圍內
        value = np.clip(value, min_val, max_val)
        
        # 計算離散化後的索引
        scaling = (value - min_val) / (max_val - min_val)
        index = int(scaling * (num_bins - 1))
        return index
        
    def _initialize_action_table(self):
        """初始化動作查找表"""
        # 創建一個多維數組來存儲動作
        table_shape = (self.pos_bins, self.vel_bins, self.ang_bins, self.angvel_bins)
        action_table = np.zeros(table_shape, dtype=int)
        
        # 根據物理規律填充表格
        for pos in range(self.pos_bins):
            for vel in range(self.vel_bins):
                for ang in range(self.ang_bins):
                    for angvel in range(self.angvel_bins):
                        # 將角度和角速度映射回連續值的中點
                        ang_continuous = self._get_continuous_value(ang, self.ang_range, self.ang_bins)
                        angvel_continuous = self._get_continuous_value(angvel, self.angvel_range, self.angvel_bins)
                        
                        # 基於物理直覺的控制規則
                        # 1. 如果桿子正在向右倒，向右移動
                        # 2. 如果桿子正在向左倒，向左移動
                        # 3. 考慮角速度的影響
                        action = 1 if (ang_continuous * 3 + angvel_continuous * 0.5) > 0 else 0
                        action_table[pos, vel, ang, angvel] = action
        
        return action_table
        
    def _get_continuous_value(self, index, value_range, num_bins):
        """從離散索引獲取連續值的中點"""
        min_val, max_val = value_range
        bin_size = (max_val - min_val) / (num_bins - 1)
        return min_val + bin_size * index
        
    def step(self, observation, reward=0, terminated=False):
        """查表獲取動作"""
        # 將觀測值離散化
        state_indices = self._discretize_state(observation)
        
        # 從表中查找動作
        action = self.action_table[state_indices]
        
        return action
        
    def close(self):
        pass
        
    def print_table_statistics(self):
        """輸出表格的統計信息"""
        total_entries = np.prod(self.action_table.shape)
        num_zeros = np.sum(self.action_table == 0)
        num_ones = np.sum(self.action_table == 1)
        
        print(f"Table shape: {self.action_table.shape}")
        print(f"Total entries: {total_entries}")
        print(f"Action 0 (left): {num_zeros} ({num_zeros/total_entries*100:.1f}%)")
        print(f"Action 1 (right): {num_ones} ({num_ones/total_entries*100:.1f}%)")

```

這個查表式控制器的主要特點：

1. **狀態空間離散化**：
   - 將位置分為3個區域
   - 將速度分為3個區域
   - 將角度分為6個區域（更精細）
   - 將角速度分為6個區域（更精細）

2. **動作表設計**：
   - 根據物理直覺設置預定的動作
   - 主要考慮角度和角速度的組合
   - 位置和速度作為輔助參考

3. **查表邏輯**：
   - 對輸入狀態進行離散化
   - 使用離散化的索引查找對應動作
   - 自動處理超出範圍的情況

4. **特點**：
   - 實現簡單，執行速度快
   - 不需要在線計算
   - 控制策略直觀易懂

5. **局限性**：
   - 離散化可能導致精度損失
   - 表格大小隨狀態變量增加呈指數增長
   - 固定的控制策略，缺乏適應性

這種方法的優勢在於：
1. 計算負擔極小
2. 響應時間確定
3. 容易理解和調試
4. 適合嵌入式系統

你可以通過：
1. 調整離散化的區間數
2. 修改動作表的初始化規則
3. 改變狀態空間的範圍
來優化控制效果。