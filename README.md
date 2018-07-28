# TextClassification
TCZoo

数据格式更改为：  
**train**:[train_input, label]  
**test**:test_input

均为keras标准输入，形状类似于[samples, max_len]  
**samples**代表样本总数，训练集和测试集均为**102277**  
**max_len**代表每篇文章最大长度，由于在总数据集上统计发现长度不大于1980的文章总数就有98%，因此取最大长度为2000

**label**已经one-hot化，可以直接进行分类，总类别数为19
  
