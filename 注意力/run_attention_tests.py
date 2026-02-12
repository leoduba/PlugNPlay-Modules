"""
专门测试 mla 模块下的 5 个注意力实现：
1. ScaledDotProductAttention (基础缩放点积)
2. MultiHeadAttention (标准MHA)
3. MultiQueryAttention (MQA)
4. GroupedQueryAttention (GQA)
5. MultiHeadLatentAttention (MLA)

核心测试维度：输出正确性、前向速度、内存占用
适配 Kaggle Notebook + CPU 环境
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# ===================== Kaggle 环境适配 =====================
# 克隆仓库（如果未存在）
MLA_REPO_PATH = "/kaggle/working/mla-pytorch"
if not os.path.exists(MLA_REPO_PATH):
    print("克隆 mla-pytorch 仓库...")
    os.system(f"git clone https://github.com/sshkhr/mla-pytorch.git {MLA_REPO_PATH}")

# 添加模块路径
sys.path.insert(0, MLA_REPO_PATH)

# 导入核心注意力模块
try:
    from mla.mha import MultiHeadAttention
    from mla.mla import MultiHeadLatentAttention
    from mla.mqa import MultiQueryAttention
    from mla.gqa import GroupedQueryAttention
    from mla.sdpa import ScaledDotProductAttention
    print("✅ 成功导入所有注意力模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ===================== 全局配置 =====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16 if DEVICE.type == 'cuda' else torch.float32
print(f"\n测试环境: Device={DEVICE}, Dtype={DTYPE}")

# 统一测试参数（保证公平对比，CPU下降低参数避免过慢）
TEST_CONFIG = {
    "batch_size": 4 if DEVICE.type == 'cpu' else 8,
    "seq_len": 512 if DEVICE.type == 'cpu' else 1024,
    "d_model": 512 if DEVICE.type == 'cpu' else 1024,       # 模型维度
    "n_heads": 8 if DEVICE.type == 'cpu' else 16,           # 查询头数
    "n_kv_heads_gqa": 2 if DEVICE.type == 'cpu' else 4,     # GQA的KV头数
    "d_c": 128 if DEVICE.type == 'cpu' else 256,            # MLA的KV隐空间维度
    "d_cq": 256 if DEVICE.type == 'cpu' else 512,           # MLA的Query隐空间维度
    "max_seq_len": 1024 if DEVICE.type == 'cpu' else 2048   # KV缓存最大长度
}
print(f"测试参数: {TEST_CONFIG}")

# ===================== 工具函数 =====================
def measure_memory_usage(func, *args, **kwargs):
    """测量函数执行的内存占用（仅CUDA）"""
    if DEVICE.type != 'cuda':
        return 0.0
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
    
    # 执行函数
    _ = func(*args, **kwargs)
    
    torch.cuda.synchronize()
    end_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
    return end_mem - start_mem

def benchmark_forward_speed(model, input_data, warmup_runs=3, test_runs=5 if DEVICE.type == 'cpu' else 10):
    """测量前向传播速度（平均耗时），CPU下减少测试次数"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*input_data)
    
    # 正式测试
    torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(test_runs):
            _ = model(*input_data)
    
    torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
    total_time = time.perf_counter() - start_time
    
    avg_time = total_time / test_runs
    tokens_per_sec = (TEST_CONFIG["batch_size"] * TEST_CONFIG["seq_len"]) / avg_time
    return avg_time * 1000, tokens_per_sec  # 毫秒/次, token/秒

# ===================== 初始化所有测试模块 =====================
def init_attention_modules():
    """初始化所有注意力模块（统一参数）"""
    modules = {}
    
    # 1. ScaledDotProductAttention (基础模块，适配其输入形状要求)
    modules["ScaledDotProductAttention"] = {
        "model": ScaledDotProductAttention().to(DEVICE, dtype=DTYPE),
        "input_type": "qkv"  # 输入为 (batch, seq_len, d_k) 形状的Q/K/V
    }
    
    # 2. MultiHeadAttention (标准MHA)
    modules["MultiHeadAttention"] = {
        "model": MultiHeadAttention(
            d_model=TEST_CONFIG["d_model"],
            n_heads=TEST_CONFIG["n_heads"],
            max_seq_len=TEST_CONFIG["max_seq_len"]
        ).to(DEVICE, dtype=DTYPE),
        "input_type": "x"  # 输入为原始x
    }
    
    # 3. MultiQueryAttention (MQA)
    modules["MultiQueryAttention"] = {
        "model": MultiQueryAttention(
            d_model=TEST_CONFIG["d_model"],
            n_heads=TEST_CONFIG["n_heads"],
            max_seq_len=TEST_CONFIG["max_seq_len"]
        ).to(DEVICE, dtype=DTYPE),
        "input_type": "x"
    }
    
    # 4. GroupedQueryAttention (GQA)
    modules["GroupedQueryAttention"] = {
        "model": GroupedQueryAttention(
            d_model=TEST_CONFIG["d_model"],
            n_heads=TEST_CONFIG["n_heads"],
            n_kv_heads=TEST_CONFIG["n_kv_heads_gqa"],
            max_seq_len=TEST_CONFIG["max_seq_len"]
        ).to(DEVICE, dtype=DTYPE),
        "input_type": "x"
    }
    
    # 5. MultiHeadLatentAttention (MLA)
    modules["MultiHeadLatentAttention"] = {
        "model": MultiHeadLatentAttention(
            d_model=TEST_CONFIG["d_model"],
            n_heads=TEST_CONFIG["n_heads"],
            d_c=TEST_CONFIG["d_c"],
            d_cq=TEST_CONFIG["d_cq"],
            max_seq_len=TEST_CONFIG["max_seq_len"]
        ).to(DEVICE, dtype=DTYPE),
        "input_type": "x"
    }
    
    return modules

# ===================== 生成测试输入 =====================
def generate_test_inputs():
    """生成适配不同模块的测试输入（修复SDP的维度问题）"""
    # 基础输入 x: [batch, seq_len, d_model]
    x = torch.randn(
        TEST_CONFIG["batch_size"],
        TEST_CONFIG["seq_len"],
        TEST_CONFIG["d_model"],
        device=DEVICE,
        dtype=DTYPE
    )
    
    # 为ScaledDotProductAttention生成正确形状的Q/K/V: (batch, seq_len, d_k)
    # d_k = d_model / n_heads（与多头拆分后的维度一致）
    d_k = TEST_CONFIG["d_model"] // TEST_CONFIG["n_heads"]
    Q = torch.randn(
        TEST_CONFIG["batch_size"],
        TEST_CONFIG["seq_len"],
        d_k,
        device=DEVICE,
        dtype=DTYPE
    )
    K = Q.clone()  # 简化测试，K/V与Q相同
    V = Q.clone()
    
    return {
        "x": x,
        "qkv": (Q, K, V)
    }

# ===================== 核心测试流程 =====================
def run_attention_tests():
    print("\n" + "="*80)
    print("开始注意力模块对比测试")
    print("="*80)
    
    # 初始化模块 + 生成输入
    modules = init_attention_modules()
    inputs = generate_test_inputs()
    
    # 测试结果存储
    results = {
        "module": [],
        "output_shape": [],
        "forward_time_ms": [],
        "tokens_per_sec": [],
        "memory_mb": [],
        "param_count": []
    }
    
    # 逐个测试模块
    for name, info in modules.items():
        print(f"\n--- 测试 {name} ---")
        model = info["model"]
        
        # 1. 生成适配输入
        if info["input_type"] == "x":
            input_data = (inputs["x"],)
        else:  # qkv (适配SDP的正确维度)
            input_data = inputs["qkv"]
        
        # 2. 前向传播（验证输出正确性）
        with torch.no_grad():
            output = model(*input_data)
        
        # 处理输出形状（统一格式）
        if name == "ScaledDotProductAttention":
            # SDP返回 (output, attn_weights)，取output
            output_shape = output[0].shape
        else:
            output_shape = output.shape
        print(f"  输出形状: {output_shape}")
        
        # 3. 测试前向速度
        forward_time, tokens_per_sec = benchmark_forward_speed(model, input_data)
        print(f"  平均前向耗时: {forward_time:.2f} ms")
        print(f"  吞吐量: {tokens_per_sec:.0f} tokens/sec")
        
        # 4. 测试内存占用
        mem_usage = measure_memory_usage(model, *input_data)
        mem_str = f"{mem_usage:.1f} MB" if DEVICE.type == 'cuda' else "N/A (CPU)"
        print(f"  内存占用: {mem_str}")
        
        # 5. 统计参数量
        param_count = sum(p.numel() for p in model.parameters())
        param_str = f"{param_count/1e6:.2f}M"
        print(f"  参数量: {param_str}")
        
        # 保存结果
        results["module"].append(name)
        results["output_shape"].append(output_shape)
        results["forward_time_ms"].append(forward_time)
        results["tokens_per_sec"].append(tokens_per_sec)
        results["memory_mb"].append(mem_usage)
        results["param_count"].append(param_count)
    
    # ===================== 结果可视化 =====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Attention Modules Comparison (CPU)", fontsize=14) if DEVICE.type == 'cpu' else fig.suptitle("Attention Modules Comparison (CUDA)", fontsize=14)
    
    # 1. 前向耗时对比
    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax1.bar(results["module"], results["forward_time_ms"], color=colors)
    ax1.set_title("Forward Time (ms)", fontsize=12)
    ax1.set_ylabel("Time (milliseconds)", fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    # 标注数值
    for i, v in enumerate(results["forward_time_ms"]):
        ax1.text(i, v + 0.5, f"{v:.2f}", ha='center', fontsize=8)
    
    # 2. 吞吐量对比
    ax2 = axes[1]
    ax2.bar(results["module"], results["tokens_per_sec"], color=colors)
    ax2.set_title("Throughput (tokens/sec)", fontsize=12)
    ax2.set_ylabel("Tokens per Second", fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    # 标注数值
    for i, v in enumerate(results["tokens_per_sec"]):
        ax2.text(i, v + 50, f"{v:.0f}", ha='center', fontsize=8)
    
    # 3. 内存占用对比（仅CUDA）
    ax3 = axes[2]
    if DEVICE.type == 'cuda':
        ax3.bar(results["module"], results["memory_mb"], color=colors)
        ax3.set_title("Memory Usage (MB)", fontsize=12)
        ax3.set_ylabel("Memory (MB)", fontsize=10)
        # 标注数值
        for i, v in enumerate(results["memory_mb"]):
            ax3.text(i, v + 5, f"{v:.1f}", ha='center', fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Memory test only for CUDA", ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        ax3.set_title("Memory Usage (MB)", fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("/kaggle/working/attention_modules_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ 对比图表已保存至: /kaggle/working/attention_modules_comparison.png")
    
    # ===================== 结果汇总表格 =====================
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    header = f"{'模块名':<30} {'输出形状':<20} {'耗时(ms)':<10} {'吞吐量(tok/s)':<15} {'内存(MB)':<12} {'参数量':<10}"
    print(header)
    print("-"*80)
    for i in range(len(results["module"])):
        module = results["module"][i]
        shape = str(results["output_shape"][i])[:18] + "..." if len(str(results["output_shape"][i])) > 20 else str(results["output_shape"][i])
        time_ms = f"{results['forward_time_ms'][i]:.2f}"
        tok_sec = f"{results['tokens_per_sec'][i]:.0f}"
        mem = f"{results['memory_mb'][i]:.1f}" if DEVICE.type == 'cuda' else "N/A"
        params = f"{results['param_count'][i]/1e6:.2f}M"
        
        row = f"{module:<30} {shape:<20} {time_ms:<10} {tok_sec:<15} {mem:<12} {params:<10}"
        print(row)

# ===================== 运行测试 =====================
if __name__ == "__main__":
    run_attention_tests()
    print("\n🎉 所有测试完成！")
    print("生成的文件:")
    print("  - /kaggle/working/attention_modules_comparison.png (对比图表)")
