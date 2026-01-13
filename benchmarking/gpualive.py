import torch
import time
import multiprocessing
import os

def gpu_worker(gpu_id, interval, matrix_size):
    """
    运行在特定 GPU 上的工作进程
    """
    # 在 spawn 模式下，每个进程都是新的，设置可见设备为当前分配的 ID
    # 这样在这个进程看来，它只有一张卡，即 cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 这里的 import 有时在 spawn 模式下需要重新确保加载，但在顶层导入通常没问题
    # 关键是不要在设置 CUDA_VISIBLE_DEVICES 之前调用任何 cuda 函数
    
    try:
        device = torch.device("cuda:0")
        print(f"[GPU {gpu_id}] 正在启动负载... (PID: {os.getpid()})")
        
        # 初始化矩阵
        x = torch.randn(matrix_size, matrix_size, device=device)
        y = torch.randn(matrix_size, matrix_size, device=device)
        
        while True:
            _ = torch.mm(x, y)
            torch.cuda.synchronize()
            if interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        pass # 子进程静默退出
    except Exception as e:
        print(f"[GPU {gpu_id}] 出错: {e}")

def run_load_on_all_gpus(interval=0.05, matrix_size=10000):
    if not torch.cuda.is_available():
        print("未检测到 GPU。")
        return

    # 在主进程获取卡数量
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 张 GPU，准备全部加载...")
    
    processes = []
    
    for i in range(gpu_count):
        p = multiprocessing.Process(target=gpu_worker, args=(i, interval, matrix_size))
        p.start()
        processes.append(p)
        
    print(f"所有 GPU ({gpu_count} 张) 负载已启动。按 Ctrl+C 停止。")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n正在停止所有负载进程...")
        for p in processes:
            p.terminate()
            p.join()
        print("已停止。")

if __name__ == "__main__":
    # --- 关键修复 ---
    # 强制使用 'spawn' 方式启动子进程，解决 CUDA 初始化报错问题
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    # ----------------
    
    # 运行负载
    run_load_on_all_gpus(interval=0.02, matrix_size=12000)