import numpy as np

class FuzzyAgent:
    def __init__(self, env):
        # 定義模糊集的範圍
        self.angle_range = np.pi/4  # ±45度
        self.velocity_range = 4.0    # 角速度範圍
        
        # 定義語言變量的模糊集
        NB = self.NB = -2  # Negative Big
        NS = self.NS = -1  # Negative Small
        ZE = self.ZE = 0   # Zero
        PS = self.PS = 1   # Positive Small
        PB = self.PB = 2   # Positive Big
        
        # 控制規則表 (angle × angular_velocity → action)
        self.rule_table = [
            #NB  NS   ZE   PS   PB  # angle
            [NB, NB, NS, NS, ZE],  # NB angular_velocity
            [NB, NS, NS, ZE, PS],  # NS
            [NS, NS, ZE, PS, PS],  # ZE
            [NS, ZE, PS, PS, PB],  # PS
            [ZE, PS, PS, PB, PB]   # PB
        ]
        
    def reset(self, mode=None):
        return
        
    def _fuzzify_angle(self, angle):
        """角度模糊化"""
        # 歸一化角度到 [-1, 1]
        normalized_angle = np.clip(angle / self.angle_range, -1, 1)
        
        # 計算隸屬度
        memberships = []
        points = [-1, -0.5, 0, 0.5, 1]  # 對應 NB, NS, ZE, PS, PB
        
        for point in points:
            # 使用三角形隸屬函數
            membership = max(0, 1 - abs(normalized_angle - point) * 2)
            memberships.append(membership)
            
        return memberships
        
    def _fuzzify_velocity(self, velocity):
        """角速度模糊化"""
        # 歸一化角速度到 [-1, 1]
        normalized_velocity = np.clip(velocity / self.velocity_range, -1, 1)
        
        # 計算隸屬度
        memberships = []
        points = [-1, -0.5, 0, 0.5, 1]
        
        for point in points:
            membership = max(0, 1 - abs(normalized_velocity - point) * 2)
            memberships.append(membership)
            
        return memberships
        
    def _fuzzy_inference(self, angle_memb, velocity_memb):
        """模糊推理"""
        # 初始化輸出隸屬度
        output_strength = {self.NB: 0, self.NS: 0, self.ZE: 0, 
                         self.PS: 0, self.PB: 0}
        
        # 對每個規則進行推理
        for i in range(5):  # angle
            for j in range(5):  # velocity
                # 取小值作為規則強度
                rule_strength = min(angle_memb[i], velocity_memb[j])
                # 更新輸出隸屬度
                rule_output = self.rule_table[i][j]
                output_strength[rule_output] = max(
                    output_strength[rule_output], 
                    rule_strength
                )
                
        return output_strength
        
    def _defuzzify(self, output_strength):
        """重心法解模糊化"""
        numerator = 0
        denominator = 0
        
        # 輸出值的離散點
        output_points = {
            self.NB: -1.0,
            self.NS: -0.5,
            self.ZE: 0.0,
            self.PS: 0.5,
            self.PB: 1.0
        }
        
        for output_val, strength in output_strength.items():
            point = output_points[output_val]
            numerator += point * strength
            denominator += strength
            
        if denominator == 0:
            return 0
            
        return numerator / denominator
        
    def step(self, observation, reward=0, terminated=False):
        """執行一步模糊控制"""
        # 獲取角度和角速度
        _, _, angle, angle_velocity = observation
        
        # 模糊化
        angle_memberships = self._fuzzify_angle(angle)
        velocity_memberships = self._fuzzify_velocity(angle_velocity)
        
        # 模糊推理
        output_strength = self._fuzzy_inference(
            angle_memberships, 
            velocity_memberships
        )
        
        # 解模糊化
        control = self._defuzzify(output_strength)
        
        # 轉換為離散動作
        action = 1 if control > 0 else 0
        
        return action
        
    def close(self):
        pass