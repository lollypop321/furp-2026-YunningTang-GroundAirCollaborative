# Weekly Progress Log

> Update this file **every week**. Add a new entry at the top for each week.
> This is the first thing we check during review. Keep it honest and specific — it also feeds your attendance record (Rule 1).

**How to use:** copy the *Week template* block below for each new week. Newest week goes at the top.

---

## Week template — copy me

### Week N — YYYY-MM-DD

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

**Progress this week**
- _What did you actually do / finish?_

**Challenges & blockers**
- _What got in the way? What are you stuck on?_

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._

---

<!-- =================  YOUR ENTRIES BELOW  ================= -->

### Week 3 — 2026-06-22

# 卡车-无人机协同配送问题实验报告

## 1. 实验目的

本实验旨在复现并评估一篇关于带时间窗的多卡车-无人机协同配送问题研究的核心算法——协同帕累托蚁群算法（Collaborative P-ACO），并通过在不同规模算例上的测试，分析其性能与适用性。

## 2. 实验设置

### 2.1 测试实例

本实验参考论文设定，采用随机生成的两种规模算例：

| 算例规模 | 客户数量 | 卡车数量 | 无人机数量 |
|-----|-----|-----|-----|
| 小型 | 25 | 2 | 2 |
| 中型 | 50 | 6 | 6 |

所有客户坐标在[-15, 15]范围内随机生成，时间窗[e_i, l_i]分别在[5, 30]和[60, 110]区间内随机生成。

### 2.2 算法参数

| 参数 | 符号| 值|
|-----|-----|-----|
| 信息素重要程度 | α | 1 |
| 启发值重要程度 | β | 2 |
| 探索概率 | q₀ | 0.5 |
| 全局挥发系数 | ρ | 0.15 |
| 局部挥发系数 | ξ | 0.1 |
| 蚂蚁数量 | K | 50（25客户）/ 70（50客户） |
| 最大迭代次数 | MaxIt | 100（25客户）/ 170（50客户） |

### 2.3 评估指标

- 总成本：卡车行驶距离 + 无人机飞行距离（目标函数A）
- 总延迟：所有客户的时间窗违反惩罚之和（目标函数B）
- 综合目标：总成本 + 总延迟
- 帕累托前沿：算法找到的非支配解集合

## 3. 实验结果

### 3.1 小型算例结果（25客户，2卡车，2无人机）

#### 最优解

- **总成本**: 211.95
- **总延迟**: 284.69
- **综合目标**: 496.64
- **卡车路径**：
  - 卡车1: [0, 12, 25, 11, 16, 7, 24, 9, 14, 21, 3, 8, 19, 10, 15, 0]
  - 卡车2: [0, 2, 18, 23, 22, 17, 6, 1, 13, 5, 0]
- **无人机分配**：
  - 无人机1: 客户20，起飞点25 -> 降落点10
  - 无人机2: 客户4，起飞点23 -> 降落点1
- **帕累托前沿（部分）**

  | 解编号 | 总成本 | 总延迟 | 综合目标 |
  |-----|-----|-----|-----|
  | 1 | 245.34 | 278.64 | 523.98 |
  | 10 | 211.95 | 284.69 | 496.64 |
  | 3 | 206.89 | 290.13 | 497.02 |

  完整帕累托前沿包含12个非支配解。

- **收敛曲线**

  - 前10次迭代: [521.69, 521.69, 521.69, 509.78, 509.78, 509.78, 509.78, 509.78, 509.78, 509.78]
  - 后10次迭代: [496.64, 496.64, 496.64, 496.64, 496.64, 496.64, 496.64, 496.64, 496.64, 496.64]

![25客户结果](/results/P-ACO_result_20260701_123820.png)

### 3.2 中型算例结果（50客户，6卡车，6无人机）

#### 最优解

- **总成本**: 381.08
- **总延迟**: 746.51
- **综合目标**: 1127.58

- **卡车路径**：
  - 卡车1: [0, 7, 36, 11, 15, 40, 25, 0]
  - 卡车2: [0, 2, 44, 5, 1, 28, 38, 0]
  - 卡车3: [0, 9, 24, 23, 37, 6, 35, 22, 4, 46, 34, 0]
  - 卡车4: [0, 49, 33, 14, 21, 30, 29, 3, 50, 10, 0]
  - 卡车5: [0, 47, 48, 45, 32, 41, 26, 18, 27, 0]
  - 卡车6: [0, 43, 12, 19, 42, 39, 0]

- **无人机分配**：
  - 无人机1: 客户16，起飞点7 -> 降落点40
  - 无人机2: 客户13，起飞点2 -> 降落点38
  - 无人机3: 客户17，起飞点9 -> 降落点22
  - 无人机4: 客户8，起飞点49 -> 降落点50
  - 无人机5: 客户20，起飞点47 -> 降落点41
  - 无人机6: 客户31，起飞点43 -> 降落点19

- **帕累托前沿（部分）**

  | 解编号 | 总成本 | 总延迟 | 综合目标 |
  |-------|--------|--------|----------|
  | 13    | 381.08 | 746.51 | 1127.58  |
  | 8     | 396.22 | 742.57 | 1138.79  |
  | 20    | 389.66 | 745.89 | 1135.55  |

完整帕累托前沿包含22个非支配解。

- **收敛曲线**
  - 前10次迭代: [1213.43, 1193.73, 1193.73, 1174.07, 1174.07, 1174.07, 1174.07, 1174.07, 1166.52, 1166.52]
  - 后10次迭代: [1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58]

![50客户结果](/results/P-ACO_result_20260701_124511.png)

## 4. 结果讨论

### 4.1 算法有效性分析

协同P-ACO能够在两种规模算例上找到可行解，验证了算法基本框架的正确性。
无人机利用率稳定：两种规模的每个无人机都分配了恰好一个客户，表明算法能够有效利用无人机资源。
时间窗约束管理：中型算例的总延迟（746.51）显著高于小型算例（284.69），表明随着问题规模扩大和车辆增多，时间窗约束变得更具挑战性。

### 4.2 收敛性分析

两个算例的收敛曲线均显示算法在前10-20次迭代内快速下降，随后逐渐趋于平稳，表明算法能够在较少的迭代次数内找到高质量解。

### 4.3 规模效应

| 指标 | 小型算例 | 中型算例 |
|-----|-----|-----|
| 客户数 | 25 | 50 |
| 车辆数 | 4 | 12 |
| 最优综合目标 | 496.64 | 1127.58 |
| 帕累托解数量 | 12 | 22 |

随着算例规模扩大，综合目标值呈非线性增长，这符合组合优化问题的典型特征。

### 4.4 论文结果对比

| 参数 | 实验成本 |	论文预期成本 | 实验延迟 | 论文预期延迟 |
|-----|-----|-----|-----|-----|
| n50-e0 | 306.40 | 85.1 | 744.71 | 127 |
| n50-e2 | 375.22 | 100.4 | 744.26 | 79.2 |
| n50-e4 | 375.22 | 101.6 | 744.26 | 75.4 |

表中展示 50 客户规模下，不同无人机续航参数的算法指标与文献 CP-ACO 均值对比结果。从成本指标来看，文献随续航增大总成本小幅上升，而本文各组实验成本整体偏高，且 e=2、e=4 工况成本无下降趋势；延误指标层面，文献续航提升可显著降低客户平均延误，本文三组工况总延误几乎无变化，优化效果缺失。

数值与趋势双重差异的核心原因为模型参数与目标函数未完全对标：其一，成本计算仅累加行驶距离，缺少文献的里程加权系数，造成数值量级差距；其二，无人机分配数量、飞行时序约束过紧，各实验组无人机极少参与配送，无法发挥协同优化作用；其三，延误统计维度不统一，本文统计所有客户延误总和，文献采用单客户平均延误。

为实现公平对比，后续将严格依照原文数学模型重构成本、延误计算逻辑，放宽无人机调度限制，并通过多次重复实验计算均值、标准差，使实验设置与文献完全统一。


### 4.5 局限性

- 模型存在多处简化：成本仅统计行驶距离，未采用原文加权成本；时间窗无分层惩罚，无人机时序约束简化，造成指标量级与变化趋势和基准不符。

- 无人机调度约束严苛，各实验组无人机使用率低，协同配送优势无法体现；且每组仅单次运行，缺少多次重复实验的统计结果，实验可靠性不足。

- 实验工况不全, 仅测试 2 台卡车、2 台无人机场景，缺少 4、6 台车辆对照；算例为随机生成，未使用文献标准算例，对比存在干扰因素，同时未对蚁群关键参数做敏感性分析。

## 5. 结论

本文采用 P-ACO 算法求解带时间窗卡车 - 无人机协同配送问题，针对 50 客户、三种无人机续航场景开展优化实验，算法能够输出可行配送路径与帕累托前沿解集，实现成本与延误双目标优化。与文献 CP-ACO 结果对比可见，本文方案总成本偏高，且续航提升无法有效降低延误。分析表明差异来源于成本函数、无人机调度约束、延误统计口径三方面建模不统一。本文验证了算法求解该类协同 VRP 问题的可行性，但模型尚未与基准算法完全对齐，后续修正建模细节后可实现公平对比。

下一步工作：

后续将完全复刻文献数学模型，统一成本、延误计算规则；放宽无人机分配限制，提升协同调度效果；补充全部车辆规模实验组，每组多次运行计算统计指标；开展参数敏感性实验，进一步优化算法求解效果，并探索更大规模算例（100+客户）的求解效果。



### Week 2 — 2026-06-15

**Attended this week's meeting:** Yes

**Progress this week**
- implement E and TW constriant to the original POMO approach
- implement E constraint to GA approach
- recreate the OR-Tools method and add ALNS hybrid solution to it
- run the 4 approaches with the same instances and compare the results

**Challenges & blockers**

1. **POMO**:
- Challenge: Model architecture modification (input features 3D→6D, decoder context)
- Solution: Extended encoder, added penalty annealing

2. **GA**:
- Challenge: Feasibility-preserving genetic operators
- Solution: Hard constraint mask (conservative energy estimation)

3. **OR-Tools**:
- Challenge: EV energy constraint not natively supported
- Solution: Approximated with capacity constraints (incomplete)

4. **ALNS**:
- Challenge: Balancing exploration vs. constraint satisfaction
- Solution: Soft constraint tolerance (violation < 5.0)

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** 

**Links (optional):** 


### Week 1 — 2026-06-08

**Attended this week's meeting:** Yes

**Progress this week**
- Set up repository from the FURP template.
- Read OR-Tools routing guides from the website
- Complete the environment configuration
- Pass the smoke test
- Establish a baseline implementation

**Challenges & blockers**
- No major blockers encountered this week

**Next steps**
### Next steps
- Extend the smoke test script to a larger VRP instance to test solver scalability
- Add basic vehicle capacity constraints to the model, following OR-Tools documentation examples
- Implement and validate the baseline model with clear, reproducible output

**Hours spent (optional):**

**Links (optional):**
- OR-Tools website: https://developers.google.com/optimization/routing/vrp?hl=zh-cn
