from bdd import BDD

def eval(bdd, result, var_values):
    eval_result = bdd.eval(result, var_values)
    print(f"當 x=True, y=False, z=True 時，布林值為: {eval_result}")

def get_all_vars(bdd, node):
    vars = set()
    _get_all_vars(bdd, node, vars)
    var_list = list(vars)
    var_names = []
    for v in var_list:
        name = bdd.reverse_var_map[v]
        var_names.append(name)
    return var_list, var_names

def _get_all_vars(bdd, node, vars):
    if node == bdd.one or node == bdd.zero:
        return
    vars.add(node.var)
    _get_all_vars(bdd, node.low, vars)
    _get_all_vars(bdd, node.high, vars)

def _sat(bdd, node, var_names, var_values):
    # print(f'var_values={var_values}')
    if len(var_values) == len(var_names):
        r = bdd.eval(node, dict(zip(var_names, var_values)))
        # print('r=', r)
        return var_values.copy() if r else None

    # 最後補 False 
    var_values.append(False)
    r0 = _sat(bdd, node, var_names, var_values)
    var_values.pop()
    if r0: return r0
 
    # 最後補 True
    var_values.append(True)
    r1 = _sat(bdd, node, var_names, var_values)
    var_values.pop()
    if r1: return r1

    return None

def sat(bdd, node):
    var_list, var_names = get_all_vars(bdd, node)
    print('var_names=', var_names)
    var_values = []
    return _sat(bdd, node, var_names, var_values)

def main():
    # 指定变量顺序（可选）
    bdd = BDD(var_order=['x', 'y', 'z'])
    
    # 创建变量
    x = bdd.create_var('x')
    y = bdd.create_var('y')
    z = bdd.create_var('z')
    
    # 布尔函数：(x AND y) OR (y AND z)
    result = bdd.apply_or(
        bdd.apply_and(x, y),   # x AND y
        bdd.apply_and(y, z)    # y AND z
    )
    print('result', result)
    
    print("布尔函数：(x AND y) OR (y AND z) 的 BDD 表示：")
    bdd.print_node(result)

    eval(bdd, result, {'x': True, 'y': False, 'z': True})
    eval(bdd, result, {'x': True, 'y': True, 'z': True})
    eval(bdd, result, {'x': False, 'y': True, 'z': True})

    var_list, var_names = get_all_vars(bdd, result)
    print(f'var_list={var_list} var_names={var_names}')
    r = sat(bdd,result)
    print('r=', r)
    if r:
        print(dict(zip(var_names, r)))

if __name__ == "__main__":
    main()
