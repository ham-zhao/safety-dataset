# 交互记录

## 阶段一：环境搭建 + 数据获取 + 探索

### 交互 1: 项目初始化
- **类型**：指令确认
- **用户输入**：启动阶段一，创建项目结构、配置文件和 README
- **处理要点**：创建完整目录结构（configs/data/notebooks/src/scripts/results/docs），生成 run_config.yaml（smoke_test/full_run 两档）、api_config.yaml（Anthropic API）、eval_config.yaml（评估+消融）、text_cleaning.yaml 和 image_text_cleaning.yaml（DJ 清洗配置）、requirements.txt、setup.sh、README.md
- **关键发现**：项目从空目录开始，无历史代码依赖

### 交互 2: 执行阶段一剩余步骤
- **类型**：指令确认
- **用户输入**：确认执行阶段一全部步骤
- **处理要点**：1) 环境安装成功（PyTorch MPS 可用，tesseract 安装）；2) 数据下载 8/8 成功（3 个 gated 数据集用合成替代）；3) 格式转换完成（文本 9,480 条 + 多模态 500 条）；4) Notebook 00/01/02 创建并验证通过

### 踩坑记录 1: HuggingFace Gated Dataset 认证
- **报错信息**：Dataset 'allenai/wildguardmix' is a gated dataset on the Hub
- **原因**：WildGuardMix、WildJailbreak、HarmBench 需要 HF 登录后申请访问权限
- **解决方案**：为每个 gated 数据集实现基于论文描述的合成替代数据生成，确保全流程可运行
- **预防建议**：提前 `huggingface-cli login` 并在 HF 网站申请数据集访问权限

### 踩坑记录 2: SafeBench 数据格式异常
- **报错信息**：category 字段存的是完整的种子问题而非类别名
- **原因**：SafeBench 数据集设计将种子问题作为列名/类别标识
- **解决方案**：实现 `_infer_safebench_category()` 关键词推断函数，从文本内容推断风险类别
- **预防建议**：下载后先 `head -1` 检查实际数据格式

### 踩坑记录 3: XSTest 多文件重复 + 标签误判
- **报错信息**：所有 2700 条样本都被标记为 harmful
- **原因**：XSTest 有 6 个 JSONL 文件（各模型评测结果），导致重复；且 type 字段（homonyms）被误判为 unsafe
- **解决方案**：只读 prompts.jsonl 并去重，XSTest prompt 默认为安全的对比数据
- **预防建议**：先用 `ls` 检查数据目录结构，理解数据集的文件组织方式

### 踩坑记录 4: Python 3.9 f-string 限制
- **报错信息**：SyntaxError: f-string: unmatched '['
- **原因**：Python 3.9 不支持 f-string 中嵌套相同类型的引号
- **解决方案**：将嵌套表达式提取为变量后再放入 f-string
- **预防建议**：避免在 f-string 中使用复杂的字典/列表索引

## 阶段二：Data-Juicer 清洗 + OCR + CLIP

### 交互 3: 确认开始阶段二
- **类型**：指令确认
- **用户输入**：确认开始第二阶段
- **处理要点**：1) 创建 4 个清洗模块（text_safety_pipeline/multimodal_pipeline/ocr_extractor/cross_modal_validator）；2) 运行清洗脚本验证通过；3) 创建并执行 Notebook 03（文本清洗）和 04（OCR+CLIP+版权）
- **关键发现**：文本清洗保留率 68.9%（去重是主要过滤源，去掉 2942 条重复），多模态合成数据因模板重复去重率高属预期

### 踩坑记录 5: Notebook JSON 中文引号转义
- **报错信息**：NotJSONError / Expecting ',' delimiter
- **原因**：直接用 Python 字符串写 Notebook JSON 时，中文引号""破坏 JSON 结构
- **解决方案**：用 Python 脚本 (create_notebooks.py) 通过 json.dump 生成 Notebook，避免手动 JSON 转义
- **预防建议**：复杂 Notebook 一律用 Python 脚本 + json.dump 生成

### 踩坑记录 6: Notebook source 行分割格式
- **报错信息**：SyntaxError: invalid syntax（所有行被拼成一行）
- **原因**：source.split("\n") 后每行末尾缺少换行符 \n
- **解决方案**：实现 _split_source() 函数，每行末尾添加 \n（最后一行除外）
- **预防建议**：Jupyter notebook source 格式要求每行以 \n 结尾

## 阶段三：数据增强 + 模型训练 + 评估 + 消融

### 交互 4: 开始阶段三
- **类型**：指令确认
- **用户输入**：开始阶段 3
- **处理要点**：
  1) **数据增强模块**（5 个）：contrastive_generator（对比样本）、category_balancer（稀缺类别）、typographic_attack（印刷术攻击）、copyright_embedding（版权 IP）、synthetic_rephraser（规则式改写）
  2) **训练模块**（3 个）：text_classifier（DistilBERT）、multimodal_classifier（CLIP Head）、training_utils
  3) **评估模块**（4 个）：safety_metrics、benchmark_runner、copyright_detector、ablation_runner
  4) **脚本**（5 个）：run_augmentation、run_training、run_evaluation、run_ablation、generate_report
  5) **Notebook 05-10** 全部创建并执行通过
  6) **最终输出文件**：safety_sft_mix.jsonl（6,772 条）、safety_dpo_pairs.jsonl（4,225 条）、safety_eval.jsonl（1,305 条）、dataset_card.md

- **关键数据**：
  - 增强数据：243 条（对比 100 + 类别 13 + 印刷术 30 + 版权 50 + 改写 50）
  - 文本分类器：DistilBERT 在 smoke_test 模式下 1 epoch 训练完成
  - 多模态分类器：CLIP ViT-B-32 特征 + 线性头训练完成
  - 消融实验：6 组（Full, -Safety, -Contrastive, -Augmentation, -Copyright, -ToxiGen）全部完成
  - 可视化：23 个 PNG 图表
  - 8 个模型 checkpoint 保存在 results/models/
