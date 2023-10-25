def two_sum(nums, target):
    # 创建一个字典来存储每个数字对应的索引
    num_dict = {}
    
    # 遍历数组
    for i, num in enumerate(nums):
        # 计算需要的补数
        complement = target - num
        
        # 如果补数在字典中，返回补数的索引和当前数字的索引
        if complement in num_dict:
            return [num_dict[complement], i]
        
        # 否则将当前数字和索引存入字典
        num_dict[num] = i

# 测试样例
nums1 = [2, 7, 11, 15]
target1 = 9
result1 = two_sum(nums1, target1)
print(nums1, result1)  # 应该输出 [0, 1]

nums2 = [3, 2, 4]
target2 = 6
result2 = two_sum(nums2, target2)
print(nums2, result2)  # 应该输出 [1, 2]

nums3 = [3, 3]
target3 = 6
result3 = two_sum(nums3, target3)
print(nums3, result3)  # 应该输出 [0, 1]

# 测试一个较大的数组，多个解
nums4 = [4, 7, 2, 8, 1, 5]
target4 = 9
result4 = two_sum(nums4, target4)
print(nums4, result4)  # 应该输出 [0, 3] 或 [1, 4]

# 测试一个较大的数组，无解
nums5 = [1, 2, 3, 4, 5, 6]
target5 = 20
result5 = two_sum(nums5, target5)
print(nums5, result5)  # 应该输出 None 或 []

# 测试负数
nums6 = [10, -5, 3, -2, 7, -8]
target6 = -2
result6 = two_sum(nums6, target6)
print(nums6, result6)  # 应该输出 [1, 3]

# 测试边界情况，只有两个元素
nums7 = [1, 2]
target7 = 3
result7 = two_sum(nums7, target7)
print(nums7, result7)  # 应该输出 [0, 1]

# 测试所有元素相同
nums8 = [7, 7, 7, 7, 7, 7]
target8 = 14
result8 = two_sum(nums8, target8)
print(nums8, result8)  # 应该输出 [0, 1]

# 测试空数组
nums9 = []
target9 = 5
result9 = two_sum(nums9, target9)
print(nums9, result9)  # 应该输出 None 或 []

# 测试目标值为0
nums10 = [-1, 0, 1, 2]
target10 = 0
result10 = two_sum(nums10, target10)
print(nums10, result10)  # 应该输出 [0, 2]

# 测试目标值为正无穷
nums11 = [1, 2, 3, 4]
target11 = float('inf')
result11 = two_sum(nums11, target11)
print(result11)  # 应该输出 None 或 []

# 测试目标值为负无穷
nums12 = [1, 2, 3, 4]
target12 = float('-inf')
result12 = two_sum(nums12, target12)
print(result12)  # 应该输出 None 或 []
