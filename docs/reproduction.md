# 复现说明

## 安装与主入口

环境采用 Python 3.12+，依赖见根目录 `requirements.txt`。已在 Python 3.13、NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2 的本地环境执行核验。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

程序按步骤写入 `outputs/step_*.txt`，全部完成后生成 `outputs/reproduction.json`。默认用第三问 3 场配对作为快速复现；`--full` 执行完整 300 场配对。第四问固定场景包含计算量较大的后验积分与站位优化，通常是快速流程中耗时最多的一步。

| 步骤 | 输入 | 输出 | 验收 |
|---|---|---|---|
| 第一问 | 内置解析案例与固定随机种子 | 几何验证结果 | 空集、无界、退化、直径、包围圆和独立裁剪检查通过 |
| 统计汇总 | 正式成绩 CSV、第三问逐场 CSV、第四问分层 CSV | `results/summary.json` | 配对一致，清除数一致，分层区间与保留结果一致 |
| 第二问 | 后验模型与已验证矩形块 | `outputs/q2/` | 重建 5 档安全圆，新增推荐点期望计算 |
| 第三问 | 固定场景索引及策略配置 | `outputs/q3/` | 每场全部清除、时间与保留值一致、输出中位场轨迹 |
| 第四问 | 固定合成场景与 C20 组合策略 | `outputs/q4/replay.json` | 全部清除、时间与保留值一致 |
| 展示图 | 保留的表格与精简轨迹 | `results/figures/` | 五组图重建成功 |

## 已完成的核验

2026 年 10 月 8 日，完整主流程六步全部通过，本机总用时约 106 秒。第三问 600 次策略执行逐场匹配保留结果，第四问固定场景完成 13/13 源清除。另行执行的 48 套定向布局整数覆盖检查全部通过；第二问的标量/批量几何、条件矩和源见证加速检查通过，最大数值差约 1.11×10⁻¹⁰。运行时间取决于设备，任务虚拟耗时按题目规则计算。

## 分模块运行

```bash
python src/q1.py --self-test
python scripts/summarize.py
python src/q2/evaluate.py
python src/search/evaluate.py --n 300
python src/directional/evaluate.py
python scripts/check_coverage.py
python scripts/figures.py
```

执行第三问时，`--start 30000 --n 300` 对应保留的配对场景；换一个起始索引可生成新合成场景。`--output` 可以指定回放目录。

## 第二问完整研究流程

快速入口重建随仓库保留的候选内域与安全圆，适合验证主要结果和学习实现。完整研究入口包含推荐点搜索、划块、独立验证三个阶段：

```bash
python src/q2/run_expected.py --scenario standard --model taper --phase search
python src/q2/run_expected.py --scenario standard --model taper --phase freeze --budget 16000
python src/q2/run_expected.py --scenario standard --model taper --phase validate
```

输出写入 `outputs/q2`。开发样本种子为 61014，参考点验证为 72015，区域验证为 83016。验证样本分四级增加至 2097152；开发划块预算控制区域细化程度。保存的候选内域包含赛时多轮细化结果，重新用较小预算划块会得到较粗的内域；完整逻辑与输入均在代码中保留。

区域验证要求新输出目录中的 `*_validated_half.csv` 不存在，防止把不同配置下的块混在一起。要重做该阶段，先将前一次 `outputs/q2` 重命名归档。开发划块可用 `--resume-partition` 延续已有 partition。

```bash
# posterior / feedback / integration / block-envelope checks
python src/q2/validate_expected.py
python src/q2/validate_acceleration.py
python src/q2/validate_extra.py
```

几何内核检测到 `clang++` 后自动编译；设定 `Q2_PUREPY=1` 可使用纯 Python 路径。加速模块只替换数值内循环，两条实现可交叉比较。

## 输出管理

`data/` 是保留的中间输入，`results/` 是可展示结果，`outputs/` 是每次执行生成的工作输出。根目录 `.gitignore` 排除环境、编译产物、缓存和运行输出，Git 中只保留可读表格、配置、源代码与展示图。

本地仿真使用固定随机流生成场景和空间误差。这部分哈希运算用于定义稳定随机数，与文件校验清单不同；仓库没有 parquet、文件哈希清单或赛时运行凭证。
