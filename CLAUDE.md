# Safety Dataset 项目规则

## 项目概况

- 多模态内容安全审核系统（TikTok 审核场景）
- 4 阶段：数据获取 → DJ 清洗 → 增强/训练/评估 → 文档/Git
- GitHub: https://github.com/ham-zhao/safety-dataset

## 项目结构约定

- 配置文件在 `configs/`，支持 `smoke_test` 和 `full_run` 两档
- 数据流向：`data/raw/` → `data/unified/` → `data/cleaned/` → `data/augmented/` → `data/final/`
- Notebook 编号 00-10，对应项目各阶段
- 源代码按功能分：`src/cleaning/`、`src/augmentation/`、`src/training/`、`src/evaluation/`

## 关键技术决策

- 3 个 gated HF 数据集（WildGuardMix/WildJailbreak/HarmBench）使用合成替代数据
- 文本分类器：DistilBERT（AUC=0.9962）
- 多模态分类器：CLIP ViT-B-32 冻结 + 线性头（AUC=0.9506）
- 消融实验：6 组对比验证每个数据组件的贡献

## 执行清单

每次修改代码后：
1. 相关脚本必须重新执行
2. 相关 notebook 必须 `nbconvert --execute --inplace` 重跑
3. 检查 `results/` 下输出文件是否更新
