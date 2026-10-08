# 数据与代码导航

## 建议阅读顺序

先读 `src/q1.py` 的 `constraints → solve → minimum_enclosing_circle`，再读第二问 `bayes.samples → expectation.envelopes → run_expected.validate_regions → expected_cores.cores`。理解单源几何与选点后，进入第三问 `Robot → RoutingRobot → LeanGradientRobot`；第四问从 `strategy.make_robot` 和 `policy.DirectionalBelief` 进入覆盖、后验与排路分支。

## 核心模块

| 位置 | 职责 |
|---|---|
| `src/q1.py` | 第一问的独立几何求解和自检 |
| `src/q2/geometry.py` | 首测坐标系、反馈可行集、连续几何上下包络 |
| `src/q2/bayes.py` | 位置与接收半径条件采样；直接蒙特卡洛参考 |
| `src/q2/expectation.py` | 对第二次误差、半径求条件积分与置信区间 |
| `src/q2/run_expected.py` | 开发选点、四叉树划块、独立区域验证 |
| `src/q2/expected_cores.py` | 内域栅格化、距离变换和可选圆提取 |
| `src/q2/fast*`, `batch*`, `moments*`, `source_witness*` | 数值计算的可选 C++ 内核及 Python 包装 |
| `src/q2/vendor/` | 从原支撑材料保留的底层扇区几何；上层沿用其接口 |
| `src/search/geometry.py` | 频道状态、保守多边形、失败圆与覆盖检查 |
| `src/search/robot.py` | 动作回执、时间账本、基础测量清除与兜底 |
| `src/search/routing.py` | 开放路径与搜索清除联合任务 |
| `src/search/short_baseline.py` | 格点布局、垂直探针、距离估计与射线恢复 |
| `src/search/approach.py`, `events.py`, `combined.py`, `selective_scan.py`, `gradient.py`, `station_probes.py` | 按职责分层的接近、重规划、频道选择与短基线基础逻辑 |
| `src/search/scene.py`, `simulator.py` | 固定场景与误差场、动作执行和计时 |
| `src/directional/strategy.py` | C20 组合策略的入口及局部恢复 |
| `src/directional/certify_layout_integer.py` | 整数四叉树定向覆盖校验 |
| `src/directional/uniform_field_posterior.py` | 共享均匀节点误差的 Sobol 积分后验 |
| `src/directional/correlated_bearing_posterior.py` | 插值权重、相关高斯近似及后验回退 |
| `src/directional/signal_history_likelihood.py` | 共用半径、方向对正负接收历史求似然 |
| `src/directional/nonzero_population.py` | 根据发现历史更新未知源数与类型分布 |
| `src/directional/regional_route.py`, `route_optimizer.py` | 当前任务与未来发现的联合路线 |
| `src/directional/route_anchor_choice.py`, `fixed_end_probe_order.py` | 补测格点选择与探针局部排序 |
| `src/directional/convex_station_slide.py` | 带重新覆盖验证的站位微调 |
| `scripts/` | 一键复现、统计汇总、覆盖校验及展示图 |

## 数据字典

| 数据 | 一行或一个条目的含义 | 主要字段 |
|---|---|---|
| `formal_results.csv` | 一场正式测试 | 问题、场次、清除数、每源时间、程序运行时间 |
| `q2/*_cells.csv` | 一个第二测点矩形块 | xmin、ymin、xmax、ymax、期望直径下界与上界；米制 |
| `q2/*_regions.json` | 一个场景和后验模型的内外域汇总 | 参考点、阈值、内域面积、待定面积、验证种子 |
| `q2/*_cores.json` | 每档阈值的可选圆 | 冗余、圆心、半径下界、栅格分辨率 |
| `q3/paired_300.csv` | 某个场景中一个策略的执行 | index、method、source_count、virtual_time_s、分项操作次数 |
| `q3/replay_selection.json` | 展示轨迹的选场规则 | 300 场中按完整方法每源耗时排序的下中位场，index=30280 |
| `q3/trace_*.csv` | 一次测量或清除动作 | step、action、channel、x、y、result、累计虚拟时间 |
| `q4/episodes.csv` | 历史分层比较的一场策略执行 | group、arm、case、N、D、cleared、seconds_per_source、五项成本 |
| `q4/scene.json` | 一个固定的合成场景 | 源位置、方向、半径与合成误差种子；由评估器读取 |
| `q4/trace_directional.csv` | 定向策略固定场景的动作 | 与第三问轨迹字段一致 |

第三问保留 `baseline`、`shared_probe`、`short_baseline` 三种历史策略结果；主复现入口重新执行基准和最终短基线两种策略。第四问历史逐场表用于统计复算，当前策略完整回放以 `scene.json` 为入口。

## 核心数据流

```text
场景与误差场 → 模拟器 → 普通测量/清除回执
                           ↓
             频道可行区域 + 观测历史 + 清除状态
                           ↓
           覆盖检查 / 位置推断 / 下一任务选择
                           ↓
                   移动、检测、清除
                           ↓
                 累计成本与结束条件
```

策略通过 `transport(path, request)` 调用环境；评估器在环境侧保存源真值，用于执行动作与计算指标。这个接口分工使策略层与场景生成层可以独立替换。
