# Notebook 中文说明

这个文档只解释 3 个 notebook 里面 `code` 类型的 cell 在做什么，不解释 `markdown` cell。

对应的 3 个 notebook 是：

- [01_preprocess.ipynb](D:\15_MAI\pynotebook\nlpProject\01_preprocess.ipynb)
- [02_multitask_model.ipynb](D:\15_MAI\pynotebook\nlpProject\02_multitask_model.ipynb)
- [03_train_eval.ipynb](D:\15_MAI\pynotebook\nlpProject\03_train_eval.ipynb)

## 01_preprocess.ipynb

这个 notebook 的作用是：

- 读取原始数据集
- 做标签处理和格式统一
- 保存后续训练使用的处理结果

### Cell 2

作用：
- 导入基础库：
  - `Path`
  - `json`
  - `pandas`
- 定义项目里的主要路径：
  - `PROJECT_DIR`
  - `DATA_DIR`
  - `SEMEVAL_DIR`
  - `GOEMOTIONS_DIR`
  - `PROCESSED_DIR`
- 自动创建 `processed` 文件夹

为什么重要：
- 后面所有读写文件都依赖这一格。

### Cell 3

作用：
- 定义 ABSA 标签到数字的映射：
  - `positive -> 0`
  - `negative -> 1`
  - `neutral -> 2`
  - `conflict -> 3`
- 定义 emotion 标签到数字的映射：
  - `joy -> 0`
  - `anger -> 1`
  - `sadness -> 2`
  - `fear -> 3`
  - `surprise -> 4`
  - `neutral -> 5`
- 同时建立反向映射字典

为什么重要：
- 模型训练只能处理数字标签，不能直接处理字符串标签。

### Cell 5

作用：
- 定义 `load_semeval_csv(...)`
- 读取 `SemEval` 的 CSV 文件
- 统一列名：
  - `Sentence -> text`
  - `Aspect Term -> aspect`
  - `polarity -> label_name`
- 保留项目里需要的列
- 把情感标签映射成数字
- 增加：
  - `task = 'absa'`
  - `domain = laptop / restaurant`
- 读取：
  - `Laptop_Train_v2.csv`
  - `Restaurants_Train_v2.csv`
- 最后合并成一个 `semeval_df`

为什么重要：
- 这是把原始 SemEval 数据变成可训练 ABSA 样本的第一步。

### Cell 6

作用：
- 定义 `split_semeval_by_domain(...)`
- 把 `SemEval` 数据按 domain 分层切分成：
  - `train`
  - `dev`
  - `test`
- 切分时尽量保持各标签比例不变

为什么重要：
- 当前项目没有直接使用官方 gold test 文件，而是从带标签训练集里自己切分数据。

### Cell 7

作用：
- 统计 SemEval 数据在不同：
  - `domain`
  - `split`
  - `label_name`
下的数量

为什么重要：
- 检查切分结果和标签分布是否正常。

### Cell 9

作用：
- 读取 `GoEmotions` 的 `emotions.txt`
- 读取 `ekman_mapping.json`
- 构建从细粒度情绪到粗粒度情绪的映射
- 增加一条合并规则：
  - `disgust -> anger`
- 保留：
  - `neutral -> neutral`

为什么重要：
- 这个项目不是直接用原始 27+1 类，而是把它压成 6 类。

### Cell 10

作用：
- 定义几个处理 GoEmotions 的辅助函数：
  - `parse_goemotions_label_ids`
  - `label_ids_to_names`
  - `map_to_single_coarse_label`
  - `load_goemotions_split`
- 读取：
  - `train.tsv`
  - `dev.tsv`
  - `test.tsv`
- 把原始标签映射成 6 类情绪
- 丢掉：
  - 映射后还对应多个粗粒度情绪的样本
  - `label_id` 为空的样本
- 最终生成 `goemotions_df`

为什么重要：
- 这是 emotion 任务标签压缩和清洗的核心部分。

### Cell 11

作用：
- 查看处理后的 GoEmotions 数据分布
- 统计每个 split 下每种情绪的数量

为什么重要：
- 可以确认标签映射之后数据是否合理。

### Cell 12

作用：
- 单独检查 GoEmotions 训练集
- 输出：
  - 保留了多少条样本
  - 丢掉了多少条样本

为什么重要：
- 这部分结果可以写进你的实验设置或数据处理说明里。

### Cell 14

作用：
- 把 `SemEval` 和 `GoEmotions` 分别整理成统一格式
- 两个 dataframe 最终都会包含这些字段：
  - `task`
  - `domain`
  - `split`
  - `sample_id`
  - `text`
  - `aspect`
  - `label_name`
  - `label_id`

为什么重要：
- 后面的建模和训练可以用统一方式读取数据，不用分别写两套逻辑。

### Cell 15

作用：
- 输出最终 ABSA 和 emotion 数据的形状
- 展示一些随机样本

为什么重要：
- 保存前做最后一次检查，确认处理后的结果没问题。

### Cell 17

作用：
- 保存处理好的 ABSA 文件：
  - [absa_semeval_2014.csv](D:\15_MAI\pynotebook\nlpProject\data\processed\absa_semeval_2014.csv)
- 保存处理好的 emotion 文件：
  - [emotion_goemotions_6class.csv](D:\15_MAI\pynotebook\nlpProject\data\processed\emotion_goemotions_6class.csv)
- 保存标签说明文件：
  - [label_metadata.json](D:\15_MAI\pynotebook\nlpProject\data\processed\label_metadata.json)

为什么重要：
- 后面的两个 notebook 都直接依赖这里保存的文件。

## 02_multitask_model.ipynb

这个 notebook 的作用是：

- 读取处理好的数据
- 构造 tokenizer、dataset、dataloader
- 定义多任务模型
- 做前向检查

### Cell 2

作用：
- 导入模型相关的依赖：
  - `torch`
  - `nn`
  - `Dataset`, `DataLoader`
  - `BertTokenizer`, `BertModel`
- 定义：
  - `DEVICE`
  - `PROJECT_DIR`
  - `PROCESSED_DIR`
  - `MODEL_DIR`

为什么重要：
- 这是模型 notebook 的运行环境设置。

### Cell 3

作用：
- 读取处理好的两个 CSV
- 读取标签 metadata
- 恢复：
  - `ABSA_LABEL2ID`
  - `EMOTION_LABEL2ID`
  - 反向标签字典

为什么重要：
- 模型后续输出的是数字 id，需要字典才能转回标签名。

### Cell 4

作用：
- 把两个任务的数据按 `split` 分成：
  - `train`
  - `dev`
  - `test`
- 输出每个 split 的数量

为什么重要：
- 确认处理后的数据读入正确。

### Cell 6

作用：
- 设置：
  - `MODEL_NAME`
  - `MAX_LENGTH`
  - `BATCH_SIZE`
- 从本地模型目录读取 tokenizer

为什么重要：
- 决定输入文本最大长度和每个 batch 的规模。

### Cell 8

作用：
- 定义 `ABSADataset`
- 定义 `EmotionDataset`

`ABSADataset`：
- 输入是：
  - `sentence`
  - `aspect`
- 输出：
  - `input_ids`
  - `attention_mask`
  - `token_type_ids`（如果有）
  - `labels`

`EmotionDataset`：
- 输入只有：
  - `sentence`
- 输出同样的张量结构

为什么重要：
- 这是 dataframe 到 PyTorch tensor 的桥梁。

### Cell 9

作用：
- 实例化一个 ABSA dataset 和一个 emotion dataset
- 打印一个样本

为什么重要：
- 检查每个样本返回的字段和张量格式是不是对的。

### Cell 11

作用：
- 构建两个任务的：
  - train dataloader
  - dev dataloader
  - test dataloader

为什么重要：
- 后续训练和验证都靠这些 dataloader 提供 batch。

### Cell 13

作用：
- 定义 `MultiTaskBERT`

模型结构是：
- 一个共享的 `BertModel`
- 取 `[CLS]` 向量作为句子表示
- 上面接两个分类头：
  - `absa_classifier`
  - `emotion_classifier`

为什么重要：
- 这是整个项目最核心的网络结构。

### Cell 14

作用：
- 实例化多任务模型
- 把模型放到 `CPU` 或 `GPU`

为什么重要：
- 这一步确认本地 BERT 模型文件能够正常加载。

### Cell 16

作用：
- 取一个 ABSA batch
- 做一次前向传播
- 输出 `ABSA logits shape`

为什么重要：
- 用来检查 ABSA 分支是不是通的。

### Cell 17

作用：
- 取一个 emotion batch
- 做一次前向传播
- 输出 `Emotion logits shape`

为什么重要：
- 用来检查 emotion 分支是不是通的。

## 03_train_eval.ipynb

这个 notebook 的作用是：

- 构建训练流程
- 做多任务交替训练
- 在 dev/test 上评估
- 保存最优模型
- 展示预测案例

### Cell 2

作用：
- 导入训练需要的依赖
- 设置随机种子
- 定义：
  - `DEVICE`
  - `PROJECT_DIR`
  - `PROCESSED_DIR`
  - `MODEL_DIR`
  - `CHECKPOINT_DIR`

为什么重要：
- 这是训练 notebook 的初始化环境。

### Cell 3

作用：
- 读取处理好的数据
- 读取标签 metadata
- 拆分成：
  - `train`
  - `dev`
  - `test`

为什么重要：
- 后续训练、验证、测试都从这里拿数据。

### Cell 4

作用：
- 设置训练超参数：
  - `MODEL_NAME`
  - `MAX_LENGTH`
  - `BATCH_SIZE`
  - `NUM_EPOCHS`
  - `LEARNING_RATE`
- 读取本地 tokenizer

为什么重要：
- 想调速度、显存、效果时，一般先改这一格。

### Cell 6

作用：
- 在训练 notebook 里重新定义：
  - `ABSADataset`
  - `EmotionDataset`

为什么重要：
- 让这个 notebook 自己单独运行时也没问题，不依赖前面的 notebook 状态。

### Cell 7

作用：
- 构建两个任务的所有 dataloader

为什么重要：
- 训练主循环和评估函数都要从这里拿 batch。

### Cell 9

作用：
- 在训练 notebook 里重新定义 `MultiTaskBERT`

为什么重要：
- 同样是为了让这个 notebook 自包含。

### Cell 10

作用：
- 实例化模型
- 定义损失函数：
  - `absa_loss_fn`
  - `emotion_loss_fn`
- 定义：
  - `optimizer`
  - `scheduler`
- 计算总训练步数

为什么重要：
- 这一步把训练需要的优化器和调度器都准备好了。

### Cell 12

作用：
- 定义 `move_batch_to_device(...)`
- 定义 `evaluate_model(...)`

`move_batch_to_device`：
- 把 batch 中的 tensor 搬到 `CPU` 或 `GPU`

`evaluate_model`：
- 在一个 dataloader 上做推理
- 计算：
  - 平均 loss
  - accuracy
  - macro-F1
  - classification report

为什么重要：
- 所有 dev/test 的指标计算都依赖这两个函数。

### Cell 14

作用：
- 如果你的当前版本里加了：
  - `gc.collect()`
  - `torch.cuda.empty_cache()`
- 那这格就是在训练前清理内存和显存缓存

为什么重要：
- 可以减少显存残留带来的 OOM 风险。

### Cell 15

作用：
- 运行完整训练主循环
- 每个 step 交替做两件事：
  - 训练一次 ABSA
  - 训练一次 emotion
- 每个 epoch 结束后：
  - 在 ABSA dev 上评估
  - 在 emotion dev 上评估
  - 计算综合分数
  - 如果效果更好，就保存 checkpoint

为什么重要：
- 这是整个项目里最核心的训练逻辑。

### Cell 17

作用：
- 从 checkpoint 加载最优模型参数：
  - [best_multitask_bert.pt](D:\15_MAI\pynotebook\nlpProject\checkpoints\best_multitask_bert.pt)

为什么重要：
- 最终测试应该用 dev 上最好的模型，而不是最后一个 epoch 的模型。

### Cell 19

作用：
- 在 test 集上评估：
  - ABSA
  - Emotion
- 输出详细分类报告

为什么重要：
- 这是你最终实验结果的主要来源。

### Cell 20

作用：
- 构造 `summary_df`
- 汇总最终 test 结果：
  - `task`
  - `accuracy`
  - `macro_f1`

为什么重要：
- 这是最适合拿去写报告和 PPT 的结果表。

### Cell 22

作用：
- 定义两个单样本预测函数：
  - `predict_absa(...)`
  - `predict_emotion(...)`

为什么重要：
- 后面做案例展示时需要单独对一句话做推理。

### Cell 23

作用：
- 随机从 `absa_test` 中抽 5 行样本
- 对每一行做：
  - ABSA 预测
  - emotion 预测
- 构造一个表格：
  - `text`
  - `aspect`
  - `absa_gold`
  - `absa_pred`
  - `absa_is_correct`
  - `emotion_pred`

为什么重要：
- 这个表适合做快速案例展示。

注意：
- 这里是按“样本行”抽，不是按“句子”抽
- 所以一句话如果有多个 aspect，这里可能只显示一个 aspect

### Cell 26

作用：
- 随机抽 5 个不同句子
- 对每个句子：
  - 先预测一次 emotion
  - 找到这个句子的所有 aspect
  - 每个 aspect 分别做 ABSA 预测
- 最终生成 `sentence_demo_df`

为什么重要：
- 这个版本更符合 ABSA 的真实展示需求，因为会把一个句子的所有 aspect 都列出来。

### Cell 27

作用：
- 用分组打印的方式展示 sentence-level demo：
  - 当前句子
  - emotion 预测
  - 这个句子的每个 aspect
  - ABSA gold
  - ABSA pred
  - ABSA 是否正确

为什么重要：
- 这是最适合你拿去答辩或展示案例分析的一格。

## 建议运行顺序

按下面顺序运行：

1. [01_preprocess.ipynb](D:\15_MAI\pynotebook\nlpProject\01_preprocess.ipynb)
2. [02_multitask_model.ipynb](D:\15_MAI\pynotebook\nlpProject\02_multitask_model.ipynb)
3. [03_train_eval.ipynb](D:\15_MAI\pynotebook\nlpProject\03_train_eval.ipynb)

## 最关键的 code cell

如果你只想抓重点，最关键的是这些：

- `01_preprocess`
  - Cell 5
  - Cell 10
  - Cell 14
  - Cell 17
- `02_multitask_model`
  - Cell 8
  - Cell 13
  - Cell 16
  - Cell 17
- `03_train_eval`
  - Cell 10
  - Cell 12
  - Cell 15
  - Cell 19
  - Cell 20
  - Cell 27

## 写报告时怎么概括整套流程

你可以直接这么讲：

1. 先把 `SemEval` 和 `GoEmotions` 预处理成统一格式。
2. 使用一个共享的 `BERT` 编码器。
3. 在共享编码器上接两个任务头：
   - 一个做 ABSA
   - 一个做 emotion classification
4. 训练时交替输入两个任务的 batch。
5. 用 dev 集选择表现最好的 checkpoint。
6. 最后在 test 集上报告结果，并用随机案例做定性分析。
