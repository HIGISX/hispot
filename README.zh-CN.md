**Languages / 语言:** [English](README.md) | **简体中文**

<div align="center">

<img src="assets/higisx-logo.png" alt="HiGISX" width="320"/>

### HiGISX 高智能空间计算团队

**HiSpot** — 面向城市分析的空间优化一体化 Python 框架

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-HIGISX%2Fhispot-181717?logo=github)](https://github.com/HIGISX/hispot)

[功能概览](#功能概览) · [快速开始](#快速开始) · [示例 Notebook](#示例-notebook) · [深度强化学习实验代码](#深度强化学习实验代码) · [本地运行与安装](#本地运行与安装) · [相关仓库](#related-higisx-repos-zh)

</div>

---

## 项目简介

**HiSpot**（*High Intelligence Spatial Optimization*）旨在为设施选址、覆盖服务、选址—路径等典型空间优化问题提供**可复现、可扩展**的分析流水线：从地理数据处理、数学模型建模、求解到结果可视化，尽量在同一套面向对象的 API 中完成，降低“商业 GIS + 外部求解器 + 自写胶水代码”的割裂成本。

本仓库对应论文：**《HiSpot: An Integrated and Accessible Python Framework for Spatial Optimization in Urban Analytics》**（匿名稿 `Manuscript_anonymous.docx`）。论文强调 HiSpot 将数据加工、模型表达、解的生成与制图表达贯通，并支持设施选址（FLP）、覆盖模型（如 MCLP / LSCP）、选址—路径（LRP）等范式，且可与常规地理数据结构衔接；同时引入面向空间优化的基准数据与真实路网多尺度实例，便于对比与复现实验。

> **英文摘要（论文）**  
> Spatial optimization plays a pivotal role in urban analytics, underpinning critical decisions from facility siting to logistics routing. HiSpot is introduced as an open-source Python framework that unifies data processing, model formulation, solution generation, and visualization within a cohesive, object-oriented API, supporting FLP, coverage models (MCLP/LSCP), and LRP, coupled with standard geospatial structures and benchmark datasets for reproducible research.

**关键词（论文）**：HiSpot；空间优化；一体化工作流；开源 Python；城市分析；可复现研究

---

## 功能概览

| 方向 | 说明 |
|------|------|
| **设施选址 FLP** | p-中值、p-中心、p-分散、p-Hub、无容量 / 有容量设施选址等 |
| **覆盖模型** | MCLP、LSCP、BCLP、MEXCLP 等经典覆盖与扩展形式 |
| **选址—路径 LRP** | 联合考虑设施选址与车辆路径（与 `pulp` 等求解器配合） |
| **地理可视化** | 与 GeoPandas、geoplot 等配合绘制服务分配、设施与需求点 |

### 示意配图（案例制图）

<p align="middle">
  <img src="img/LRP.png" width="32%" alt="LRP 示例"/>
  <img src="img/BCLP-pingshan.png" width="32%" alt="BCLP 坪山"/>
  <img src="img/MCLP-pingshan.png" width="32%" alt="MCLP 坪山"/>
</p>

<p align="middle">
  <img src="img/PHUB.png" width="24%" alt="p-Hub"/>
  <img src="img/pCenter-nansha.png" width="24%" alt="p-Center 南沙"/>
  <img src="img/pMedian-nansha.png" width="24%" alt="p-Median 南沙"/>
</p>

> 若本地缺少 `img/` 下配图，请从远端仓库同步资源或自行运行 Notebooks 生成制图。

---

## 快速开始

在已安装 Python 与 Jupyter 的环境中：

```bash
git clone https://github.com/HIGISX/hispot.git
cd hispot
pip install numpy pulp
# 可选：pip install geopandas geoplot matplotlib jupyter
```

随后打开 `Notebooks/` 中对应问题类型的笔记本，按单元格说明加载数据、建模与出图。

---

## 示例 Notebook

| 主题 | 链接 |
|------|------|
| p-Median | [pMedian.ipynb](Notebooks/pMedian.ipynb) |
| p-Center | [pCenter.ipynb](Notebooks/pCenter.ipynb) |
| P-Dispersion | [pDispersion.ipynb](Notebooks/pDispersion.ipynb) |
| MCLP | [MCLP.ipynb](Notebooks/MCLP.ipynb) |
| LSCP | [LSCP.ipynb](Notebooks/LSCP.ipynb) |
| BCLP | [BCLP.ipynb](Notebooks/BCLP.ipynb) |
| MEXCLP | [MEXCLP.ipynb](Notebooks/MEXCLP.ipynb) |
| 有容量 LRP | [LRP_cap.ipynb](Notebooks/LRP_cap.ipynb) |
| p-Hub | [pHub.ipynb](Notebooks/pHub.ipynb) |

---

## 深度强化学习实验代码

仓库内 **`DRL_Sover/`** 目录包含与 **MCLP** 等问题相关的深度学习 / 强化学习求解实验代码（如 `CacheFormer` 注意力模型、`train.py` / `run.py` 等），用于论文中的学习型求解与与传统/启发式方法对比。使用方式简述：

```bash
cd DRL_Sover
pip install torch tensorboard_logger  # 及项目所需其他依赖
python run.py --problem MCLP --help   # 查看完整超参数
```

具体训练配置请参考 `options.py` 与 `train.py`。

---

## 代码结构（节选）

```text
hispot/           # 优化模型与求解封装（FLP、Coverage、LRP 等）
Notebooks/        # Jupyter 示例与复现实验
data/             # 示例地理数据（多城市区县）
DRL_Sover/        # DRL / 注意力模型等实验代码
assets/           # 项目展示资源（含 HiGISX 团队标识）
```

---

## 本地运行与安装（原部署说明）

以下步骤与历史 README 一致，便于在本地复现 Notebook 与 whl 安装方式。

### 运行 Notebook

1. 克隆仓库：`git clone https://github.com/HIGISX/hispot.git`
2. `conda create -n higis python`
3. `conda activate higis`
4. `pip install jupyter` 后执行 `jupyter notebook`
5. `pip install pulp`
6. 按需安装发行包：`pip install HiSpot-0.1.0-py3-none-any.whl`（若仓库中提供该 wheel）

### 可选求解器（经 PuLP 调用）

- [GLPK](https://www.gnu.org/software/glpk/)（部分 conda 环境已带）
- [COIN-OR CBC](https://github.com/coin-or/Cbc)
- [CPLEX](https://www.ibm.com/analytics/cplex-optimizer)
- [Gurobi](https://www.gurobi.com/)

### 依赖一览

**核心**：`numpy`、`pulp`、HiSpot 包（`pip install higis` 或本地 whl）

**可选（制图）**：`matplotlib`、`geopandas`、`geoplot`

```bash
pip install higis numpy pulp
pip install matplotlib geopandas geoplot
```

---

<h2 id="related-higisx-repos-zh">相关 HiGISX 仓库</h2>

以下 **HiGISX** 组织内的应用型仓库为本项目提供场景扩展、数据与案例层面的支持，可与 HiSpot 方法论与工具链对照阅读。

| 仓库 | 说明 |
|------|------|
| [Hulatang-Location-Optimization](https://github.com/HIGISX/Hulatang-Location-Optimization) | 麻辣烫类选址优化 |
| [Convenient-Store-Location-Optimizaion](https://github.com/HIGISX/Convenient-Store-Location-Optimizaion) | 便利店选址优化（仓库名以 GitHub 上实际拼写为准） |
| [Guilin-Catering-Location-Optimization](https://github.com/HIGISX/Guilin-Catering-Location-Optimization) | 桂林餐饮选址优化 |
| [AEDNet](https://github.com/HIGISX/AEDNet/tree/master) | AED（自动体外除颤器）网络相关研究（`master` 分支入口） |

---

## 支持与反馈

如遇安装、数据路径或模型接口问题，欢迎在仓库中 **提交 Issue** 或通过 Discussions / Gitter 交流（见原仓库说明）。

---

<div align="center">

<sub>标识与设计 © HiGISX 团队 · HiSpot 框架持续维护中</sub>

</div>
