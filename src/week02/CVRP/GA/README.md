# GA Solver for CVRP with EV and TW

本文件夹包含与 POMO 代码相同问题设置的遗传算法（GA）求解器。

## 问题设置（与 POMO 一致）

- **问题规模**: n=50, n=100, n=200
- **约束**: 
  - 容量约束（Capacity）
  - 时间窗约束（Time Window）
  - 电动车能量约束（EV Energy）
- **EV 参数**:
  - battery_capacity: 100.0
  - energy_consumption_rate: 0.5

## 文件结构

```
GA/
├── problem_def.py              # 问题生成（与 POMO 一致）
├── cvrp_env.py                 # CVRP 环境（解码、评估）
├── ga_solver.py                # 遗传算法核心
├── solve_n50_with_EV_TW.py     # n=50 求解脚本
├── solve_n100_with_EV_TW.py    # n=100 求解脚本
└── solve_n200_with_EV_TW.py    # n=200 求解脚本
```

## 使用方法

```bash
cd /Users/tangfiona/POMO/NEW_py_ver/CVRP/GA

# 求解 n=50
python solve_n50_with_EV_TW.py

# 求解 n=100
python solve_n100_with_EV_TW.py

# 求解 n=200
python solve_n200_with_EV_TW.py
```

## GA 参数

| 参数 | n=50 | n=100 | n=200 |
|------|------|-------|-------|
| Population | 100 | 150 | 200 |
| Generations | 200 | 300 | 400 |
| Crossover Rate | 0.85 | 0.85 | 0.85 |
| Mutation Rate | 0.02 | 0.02 | 0.02 |
| Elitism Rate | 0.05 | 0.05 | 0.05 |

## 输出

- Average Cost: 平均行驶距离
- Average TW Violations: 平均时间窗违反次数
- Average Energy Violations: 平均能量违反次数
- Total Time: 总求解时间

## 与 POMO 的对比

| 特性 | GA (本代码) | POMO |
|------|-------------|------|
| 算法类型 | 遗传算法 | 深度强化学习 |
| 求解速度 | 慢（需进化多代） | 快（单次前向） |
| 最优性 | 中等 | 较好 |
| 训练需求 | 无需训练 | 需预训练模型 |
| 适用场景 | 小规模/对比基准 | 大规模/生产环境 |
