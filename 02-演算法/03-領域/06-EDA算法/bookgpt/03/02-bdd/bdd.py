class Node:
    def __init__(self, var=None, low=None, high=None):
        """
        BDD 节点的类
        
        :param var: 当前节点的变量索引
        :param low: 变量为 0 时的子节点
        :param high: 变量为 1 时的子节点
        """
        self.var = var
        self.low = low
        self.high = high
        self.hash_key = hash((var, low, high))

class BDD:
    def __init__(self, var_order=None):
        """
        初始化 BDD，支持变量名称到索引的映射
        
        :param var_order: 变量顺序列表，用于确定变量优先级
        """
        self.zero = Node()
        self.one = Node()
        self.node_cache = {}
        
        # 变量名称到索引的映射
        self.var_map = {}
        self.reverse_var_map = {}
        
        # 设置变量顺序
        if var_order:
            for idx, var_name in enumerate(var_order):
                self.var_map[var_name] = idx
                self.reverse_var_map[idx] = var_name
    
    def get_var_index(self, var_name):
        """
        获取变量的索引，如果不存在则自动分配
        
        :param var_name: 变量名称
        :return: 变量索引
        """
        if var_name not in self.var_map:
            # 自动分配新的最大索引
            new_index = len(self.var_map)
            self.var_map[var_name] = new_index
            self.reverse_var_map[new_index] = var_name
        return self.var_map[var_name]
    
    def create_var(self, var_name):
        """
        创建表示特定变量的 BDD 节点
        
        :param var_name: 变量名称
        :return: 变量的 BDD 节点
        """
        var_index = self.get_var_index(var_name)
        low = self.zero
        high = self.one
        return self.create_node(var_index, low, high)
    
    def create_node(self, var_index, low, high):
        """
        创建或返回已存在的 BDD 节点
        
        :param var_index: 变量索引
        :param low: 变量为 0 时的子节点
        :param high: 变量为 1 时的子节点
        :return: BDD 节点
        """
        # 简化节点
        if low == high:
            return low
        
        key = hash((var_index, low, high))
        if key in self.node_cache:
            return self.node_cache[key]
        
        node = Node(var_index, low, high)
        self.node_cache[key] = node
        return node
    
    def apply_and(self, u, v):
        """
        AND 运算
        
        :param u: 第一个 BDD 节点
        :param v: 第二个 BDD 节点
        :return: AND 运算结果的 BDD 节点
        """
        # 常数情况处理
        if u == self.zero or v == self.zero:
            return self.zero
        if u == self.one:
            return v
        if v == self.one:
            return u
        
        # 比较变量层级
        if u.var < v.var:
            low = self.apply_and(u.low, v)
            high = self.apply_and(u.high, v)
            return self.create_node(u.var, low, high)
        elif u.var > v.var:
            low = self.apply_and(u, v.low)
            high = self.apply_and(u, v.high)
            return self.create_node(v.var, low, high)
        else:  # u.var == v.var
            low = self.apply_and(u.low, v.low)
            high = self.apply_and(u.high, v.high)
            return self.create_node(u.var, low, high)
    
    def apply_or(self, u, v):
        """
        OR 运算
        
        :param u: 第一个 BDD 节点
        :param v: 第二个 BDD 节点
        :return: OR 运算结果的 BDD 节点
        """
        # 常数情况处理
        if u == self.one or v == self.one:
            return self.one
        if u == self.zero:
            return v
        if v == self.zero:
            return u
        
        # 比较变量层级
        if u.var < v.var:
            low = self.apply_or(u.low, v)
            high = self.apply_or(u.high, v)
            return self.create_node(u.var, low, high)
        elif u.var > v.var:
            low = self.apply_or(u, v.low)
            high = self.apply_or(u, v.high)
            return self.create_node(v.var, low, high)
        else:  # u.var == v.var
            low = self.apply_or(u.low, v.low)
            high = self.apply_or(u.high, v.high)
            return self.create_node(u.var, low, high)
    
    def print_node(self, node, depth=0):
        """
        列印 BDD 节点，使用变量名称
        
        :param node: 要列印的 BDD 节点
        :param depth: 节点深度
        """
        if node is None:
            return
        
        indent = "  " * depth
        if node.var is None:
            print(f"{indent}{'1' if node is self.one else '0'}")
            return
        
        # 使用反向映射获取变量名称
        var_name = self.reverse_var_map.get(node.var, f"var_{node.var}")
        print(f"{indent}var: {var_name}")
        print(f"{indent}low:")
        self.print_node(node.low, depth + 1)
        print(f"{indent}high:")
        self.print_node(node.high, depth + 1)

    def eval(self, node, var_values):
        """
        根據給定的變數值評估 BDD 表達式的布林值
        
        :param node: 要評估的 BDD 節點
        :param var_values: 包含變數值的字典，變數名作為鍵，布林值作為值
        :return: 根據 BDD 計算出來的布林值
        """
        if node == self.zero:
            return False  # 0 表示布林值 False
        if node == self.one:
            return True   # 1 表示布林值 True
        
        # 獲取變數的名稱
        var_name = self.reverse_var_map.get(node.var, f"var_{node.var}")
        
        # 根據 var_name 確定變數值
        var_value = var_values.get(var_name, None)
        
        if var_value is None:
            raise ValueError(f"未提供變數 {var_name} 的值")

        # 根據變數值決定走哪條分支
        if var_value:
            return self.eval(node.high, var_values)  # 如果變數值為 1，走 high 分支
        else:
            return self.eval(node.low, var_values)   # 如果變數值為 0，走 low 分支
