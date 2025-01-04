# Sales_Forecasting

## 项目简介
本项目旨在解决 Rossmann 零售公司在销售额预测中的难题，通过建立一个销售预测模型，预测Rossmann旗下1,115家门店未来6周的销售额。

在该项目中，我通过对历史销售数据进行深入分析，结合特征工程和模型优化技术，构建了一个基于 **LightGBM** 的销售预测模型，在 Kaggle 比赛中获得了 Top 10% 的成绩（由于该比赛已经结束，排名是参考了Kaggle LeaderBoard的历史排名）。


该项目主要包含以下几个部分：

1. 数据预处理：包括缺失值填充、归一化、特征编码等操作
2. 探索性数据分析（EDA）：观察数据分布，验证假设并指导特征选择
3. 特征工程：提取多种交叉特征和统计特征，增强模型捕捉复杂关系的能力
4. 模型训练与优化：采用线性回归和 LightGBM 模型，并通过特征选择、调参与模型融合提升性能
5. 结果评估与提交：对结果进行分析，优化最终的预测效果

**数据来源**：[Kaggle - Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data)

---

## 环境配置

1. **安装 Python**：
   请确保已安装 Python 3.7 或更高版本

2. **安装依赖包**：
   主要依赖包包括：
   - `pandas` (版本 ≥ 1.3.0)：用于数据处理
   - `numpy` (版本 ≥ 1.20.0)：数值计算
   - `matplotlib` & `seaborn`：数据可视化
   - `LinearRegression` & `Lightgbm`：核心回归模型
   - `scikit-learn`：模型评估与特征处理

3. **可选**：
   配置 GPU 支持以加速 LightGBM 训练

---
## 项目结构

```plaintext
.
├── input/                               # 数据集文件夹，包含训练集和测试集
│   ├── train.csv                        # 训练集
│   ├── test.csv                         # 测试集
│   ├── store.csv                        # 商店补充信息
│   └── sample_submission.csv            # 提交范例
├── notebooks/                           # 包含 Jupyter Notebooks 文件，进行数据分析与可视化
│   └── EDA.ipynb                        # 数据探索性分析与可视化
├── scripts/                             # Python 文件夹，用于数据预处理、特征工程、模型建立
│   ├── preprocess.py                    # 数据预处理
│   ├── extract_features.py              # 特征提取
│   ├── liner_model.py                   # 对整体数据做线性回归模型
│   ├── liner_model_each.py              # 针对不同店铺的线性回归模型
│   ├── gbm_base.py                      # 基础 GBM 模型
│   ├── gbm_feature_select.py            # 对GBM做随机特征选择
│   ├── gbm_tunning_step1.py             # 在随机特征选择阶段选择表现最好的模型，进行调参
│   ├── gbm_tunning_step2.py             # 分步调参
│   ├── gbm_tunning_step3.py             # 分步调参
│   └── gbm_ensemble.py                  # 选取表现较好的模型，做模型融合
├── output/                              # 包含预测结果
└── README.md                            # 项目说明文件
```



## 项目运行步骤

### 1. 下载数据

从 [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) 下载数据集，并将文件放入 `input/` 文件夹

### 2. 可视化探索

利用EDA.ipynb对数据进行可视化探索

### 3. 数据预处理
导入以下命令对数据进行预处理：

```bash
from preprocess import preprocess
preprocessed_df = preprocess(train, test, store)
```

该脚本将完成以下任务：
- 缺失值填充：使用中位数或合理的默认值填充缺失数据
- 归一化：对偏态特征使用 `RobustScaler` 进行标准化，其余用MinMax标准化
- 特征编码：对分类变量进行 OneHot 编码或顺序编码

### 4. 特征工程

运行以下命令提取交叉特征和统计特征：

```bash
from extract_features import extract_features
features_df = extract_features(features, preprocessed_df)
```

提取的特征共28个。


### 5. 模型训练

#### 5.1 线性回归模型
- liner_model.py ：对整体数据做线性回归模型
- liner_model_each.py：针对不同店铺的线性回归模型

### 5.2 LightGBM模型

- gbm_base.py ：根据经验选择常用的参数，生成一个LightGBM Base模型
- gbm_feature_select.py：对GBM做随机特征选择（具体来说，从GBM基准模型中，根据特征的重要性排序，选出前 16 个特征作为下一阶段 100 个模型的基础特征，剩下的 12 个特征作为随机选择的特征池；
  构建 100 个 LGM模型。这 100 个模型的特征由基础特征加上随机选择的特征构成。随机选择的规则：从 12 个特征中分别挑选 $$4,5,6,7,8$$ 个特征，每类挑选 20 次，刚好组成 100 个模型的特征，即特征个数为 20,21,22,23,24 的模型各20个）

- gbm_tunning_step1.py：在随机特征选择阶段，找到表现最好的模型，固定较大的学习率（0.1），对n_estimators调参
- gbm_tunning_step2.py：对树的结构进行调参
- gbm_tunning_step3.py：在前面两步调参的基础上，微调学习率等参数，增强模型泛化能力
- gbm_ensemble.py：选取表现较好的模型，做简单的模型融合



预测结果将保存在 `output/` 文件夹中，格式为 Kaggle 评分所需的 CSV 文件。



## 结果与评估

- 主要评估指标：**RMSPE**（均方根百分比误差）
- 在 Kaggle 的模型得分：
  - **Private Dataset**：RMSPE = 0.113（Top 10%）
  - **Public Dataset**： RMSPE = 0.122


## 参考资料

1. [Kaggle电商预测](https://www.cnblogs.com/majimaji/p/10265242.html)
2. [机器学习实战-Kaggle电商预测](https://blog.csdn.net/weixin_60536251/article/details/130894926)
3. [Kaggle销售额预测模型](https://github.com/wolegechu/Machine_Learning_Nanodegree/blob/master/Capstone/README.md)
4. [LightGBM调参报告](https://zhuanlan.zhihu.com/p/340270471)
5. [LightGBM调参](https://zhuanlan.zhihu.com/p/376485485)
6. [工程能力UP！LightGBM调参](https://www.cnblogs.com/PythonLearner/p/13364071.html)

