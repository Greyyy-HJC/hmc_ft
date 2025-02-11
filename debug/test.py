# %%
import torch

def test_roll_equivalence():
    # 创建一个随机测试张量 [batch_size, L, L]
    batch_size, L = 2, 4
    theta0 = torch.randn(batch_size, L, L)
    
    # 方法1: 连续两次roll
    result1 = torch.roll(torch.roll(theta0, shifts=-1, dims=1), shifts=-1, dims=2)
    
    # 方法2: 一次性roll两个维度
    result2 = torch.roll(theta0, shifts=(-1, -1), dims=(1, 2))
    
    # 检查两种方法的结果是否相同
    is_equal = torch.allclose(result1, result2)
    max_diff = torch.max(torch.abs(result1 - result2))
    
    print("测试张量形状:", theta0.shape)
    print("原始张量:\n", theta0)
    print("\n方法1结果:\n", result1)
    print("\n方法2结果:\n", result2)
    print("\n两种方法结果是否相同:", is_equal)
    print("最大差异:", max_diff.item())

# 运行测试
test_roll_equivalence()
# %%
import torch

def test_mask_repeat():
    # 创建一个小的测试掩码 [batch_size, 2, L, L]
    batch_size, L = 2, 4
    field_mask = torch.zeros((batch_size, 2, L, L), dtype=torch.bool)
    
    # 设置一些值为True，便于观察
    field_mask[0, 0, 0, 0] = True
    field_mask[1, 1, 1, 1] = True
    
    # 执行repeat操作
    field_mask_duplicate = field_mask.repeat(1, 2, 1, 1)
    
    print("原始掩码形状:", field_mask.shape)
    print("复制后掩码形状:", field_mask_duplicate.shape)
    
    print("\n原始掩码第一个样本:")
    for i in range(2):  # 遍历原始的2个通道
        print(f"Channel {i}:")
        print(field_mask[0, i])
        
    print("\n复制后掩码第一个样本:")
    for i in range(4):  # 遍历复制后的4个通道
        print(f"Channel {i}:")
        print(field_mask_duplicate[0, i])
    
    # 验证复制是否正确
    print("\n验证复制结果:")
    print("前两个通道与后两个通道是否相同:", 
          torch.all(field_mask_duplicate[:, :2] == field_mask_duplicate[:, 2:]))

# 运行测试
test_mask_repeat()
# %%
