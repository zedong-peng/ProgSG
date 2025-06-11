# ProgSG 项目架构概览

ProgSG是一个基于图神经网络的高层次综合(HLS)设计空间探索项目，旨在预测FPGA设计的性能和资源利用率。

## 项目概述

```mermaid
flowchart TD
    A["🔧 HLS C/C++代码"] --> B["📊 ProgramL图表示<br/>+ 源代码"]
    B --> C["🧠 多模态GNN+Transformer<br/>模型"]
    C --> D["📈 性能预测<br/>(延迟、资源利用率)"]
    D --> E["🎯 设计空间探索<br/>(DSE)"]
    E --> F["⚡ 最优设计配置"]
    
    G["🔧 编译指令(Pragmas)"] --> C
    
    style C fill:#e1f5fe
    style E fill:#f3e5f5
```

## 核心架构

```mermaid
flowchart TD
    subgraph "🗃️ 数据输入"
        A1["ProgramL图(.gexf)<br/>节点+边特征"]
        A2["源代码文件(.cpp)<br/>函数实现"]
        A3["编译指令配置<br/>Pragma设置"]
    end
    
    subgraph "🔄 数据预处理"
        B1["图特征编码<br/>节点类型+边类型"]
        B2["代码Token化<br/>CodeT5编码"]
        B3["Pragma嵌入<br/>MLP编码"]
    end
    
    subgraph "🧠 多模态模型"
        C1["GNN编码器<br/>(GAT/GCN/Transformer)"]
        C2["Code Transformer<br/>(CodeT5)"]
        C3["特征融合层<br/>"]
    end
    
    subgraph "📊 预测输出"
        D1["性能预测<br/>(perf)"]
        D2["资源利用率<br/>(LUT/FF/DSP/BRAM)"]
    end
    
    A1 --> B1 --> C1
    A2 --> B2 --> C2
    A3 --> B3 --> C3
    
    C1 --> C3
    C2 --> C3
    
    C3 --> D1
    C3 --> D2
    
    style C1 fill:#e8f5e8
    style C2 fill:#fff3e0
    style C3 fill:#f3e5f5
```

## 数据处理流程

```mermaid
flowchart TD
    subgraph "📁 原始数据"
        A1["dse_database/<br/>machsuite/poly"]
        A2["kernel.gexf<br/>程序图结构"]
        A3["kernel.cpp<br/>源代码"]
        A4["pragma配置<br/>编译指令"]
    end
    
    subgraph "🔧 预处理管道"
        B1["解析.gexf文件<br/>提取图结构"]
        B2["节点特征提取<br/>type + context"]
        B3["边特征提取<br/>control/data flow"]
        B4["代码tokenization<br/>CodeT5处理"]
        B5["Pragma编码<br/>参数向量化"]
    end
    
    subgraph "💾 处理后数据"
        C1["ProcessedGraphData<br/>torch_geometric.Data"]
        C2["节点特征矩阵<br/>(N_nodes, feature_dim)"]
        C3["边索引+特征<br/>(edge_index, edge_attr)"]
        C4["代码token序列<br/>(max_len, hidden_dim)"]
        C5["Pragma向量<br/>(pragma_dim,)"]
    end
    
    A1 --> B1
    A2 --> B1 --> B2
    A2 --> B1 --> B3
    A3 --> B4
    A4 --> B5
    
    B2 --> C2
    B3 --> C3
    B4 --> C4
    B5 --> C5
    
    C2 --> C1
    C3 --> C1
    C4 --> C1
    C5 --> C1
    
    style B1 fill:#e3f2fd
    style C1 fill:#e8f5e8
```

## 模型架构详解

```mermaid
flowchart TD
    subgraph "🔤 代码分支 (sequence_modeling)"
        SC1["源代码输入<br/>C/C++ functions"]
        SC2["CodeT5 Tokenizer<br/>特殊token处理"]
        SC3["CodeT5 Encoder<br/>Transformer layers"]
        SC4["代码嵌入<br/>(seq_len, bert_dim)"]
    end
    
    subgraph "📊 图分支 (graph_modeling)"
        GC1["ProgramL图输入<br/>节点+边特征"]
        GC2["GNN层堆叠<br/>GAT/GCN/TransformerConv"]
        GC3["跳跃连接<br/>JumpingKnowledge"]
        GC4["图池化<br/>Global Attention"]
        GC5["图嵌入<br/>(batch_size, hidden_dim)"]
    end
    
    subgraph "🔧 Pragma分支"
        PC1["Pragma配置<br/>编译指令参数"]
        PC2["MLP编码器<br/>参数嵌入"]
        PC3["Pragma嵌入<br/>(pragma_dim,)"]
    end
    
    subgraph "🤝 特征融合"
        FC1["多模态融合<br/>share_final_MLPs"]
        FC2["注意力机制<br/>node_token_interaction"]
    end
    
    subgraph "🎯 预测头"
        PH1["性能MLP<br/>perf预测"]
        PH2["资源MLPs<br/>util-LUT/FF/DSP/BRAM"]
    end
    
    SC1 --> SC2 --> SC3 --> SC4
    GC1 --> GC2 --> GC3 --> GC4 --> GC5
    PC1 --> PC2 --> PC3
    
    SC4 --> FC1
    GC5 --> FC1
    PC3 --> FC2
    
    FC1 --> FC2
    FC2 --> PH1
    FC2 --> PH2
    
    style SC3 fill:#fff3e0
    style GC2 fill:#e8f5e8
    style FC1 fill:#f3e5f5
    style PH1 fill:#ffebee
    style PH2 fill:#ffebee
```

## 训练流程

```mermaid
flowchart TD
    A["🔄 开始训练"] --> B["📚 加载数据集"]
    B --> C["🔀 数据分割<br/>Train/Val/Test"]
    C --> D["🏗️ 初始化模型<br/>多模态架构"]
    
    D --> E{"🔄 训练循环"}
    E --> F["📊 前向传播<br/>GNN+Transformer"]
    F --> G["📉 计算损失<br/>MSE Loss"]
    G --> H["🔄 反向传播<br/>梯度更新"]
    
    H --> I["✅ 验证阶段"]
    I --> J{"📈 性能提升?"}
    J -->|是| K["💾 保存最佳模型<br/>val_model_state_dict.pth"]
    J -->|否| L["📊 记录损失"]
    
    K --> M{"🔄 继续训练?"}
    L --> M
    M -->|是| E
    M -->|否| N["🎉 训练完成"]
    
    style D fill:#e1f5fe
    style F fill:#e8f5e8
    style K fill:#c8e6c9
```

## 推理流程

```mermaid
flowchart TD
    A["🚀 开始推理"] --> B["📋 加载训练好的模型<br/>val_model_state_dict.pth"]
    B --> C["📊 准备测试数据<br/>Test DataLoader"]
    
    C --> D["🔄 推理循环"]
    D --> E["📈 模型前向传播<br/>model.eval()"]
    E --> F["📊 获取预测结果<br/>out_dict"]
    
    F --> G["📐 计算评估指标"]
    G --> H["📋 RMSE计算<br/>回归指标"]
    G --> I["📊 资源利用率检查<br/>valid判断"]
    
    H --> J["📝 结果记录<br/>CSV导出"]
    I --> J
    J --> K{"🔄 更多数据?"}
    K -->|是| D
    K -->|否| L["📊 汇总结果<br/>最终报告"]
    
    style B fill:#e1f5fe
    style E fill:#e8f5e8
    style L fill:#c8e6c9
```

## 设计空间探索(DSE)流程

```mermaid
flowchart TD
    subgraph "🎯 DSE初始化"
        A1["📋 加载kernel配置<br/>machsuite/poly"]
        A2["🔧 编译设计空间<br/>pragma参数范围"]
        A3["🧠 加载训练好的GNN模型"]
    end
    
    subgraph "🔄 探索循环"
        B1["🎲 生成设计点<br/>pragma配置组合"]
        B2["📊 构建图数据<br/>apply_design_point"]
        B3["🔮 模型预测<br/>性能+资源利用率"]
        B4["✅ 有效性检查<br/>资源约束验证"]
        B5["📈 质量评估<br/>performance/latency"]
    end
    
    subgraph "🏆 结果管理"
        C1["📊 更新最佳结果<br/>Pareto前沿"]
        C2["💾 记录探索历史<br/>point -> result"]
        C3["📈 排序设计点<br/>quality排序"]
    end
    
    subgraph "🎯 输出结果"
        D1["🏅 Top-K最佳设计"]
        D2["📊 性能分析报告"]
        D3["💾 结果持久化<br/>pickle保存"]
    end
    
    A1 --> A2 --> A3
    A3 --> B1
    
    B1 --> B2 --> B3 --> B4 --> B5
    B5 --> C1 --> C2 --> C3
    
    C3 --> D1
    C3 --> D2
    C3 --> D3
    
    C2 --> B1
    
    style A3 fill:#e1f5fe
    style B3 fill:#e8f5e8
    style C1 fill:#f3e5f5
    style D1 fill:#c8e6c9
```

## 关键技术特性

### 🧠 多模态架构
- **图神经网络**: GAT/GCN/TransformerConv处理程序图结构
- **代码Transformer**: CodeT5编码源代码语义
- **Pragma嵌入**: MLP编码编译指令参数

### 📊 预测目标
- **性能指标**: 延迟(perf)预测
- **资源利用率**: LUT/FF/DSP/BRAM使用率
- **有效性检查**: 资源约束验证

### 🔄 训练策略
- **多目标学习**: 同时预测多个指标
- **损失函数**: MSE Loss for regression
- **模型保存**: 基于验证集性能的最佳模型选择

### 🎯 DSE算法
- **穷举搜索**: ExhaustiveExplorer
- **有效性剪枝**: 资源约束过滤
- **质量评估**: 基于性能的设计点排序

## 配置选项

### 🛠️ 主要标志位
- `subtask`: "train"/"inference"/"dse" - 选择运行模式
- `model`: "our" - 使用多模态架构
- `sequence_modeling`: True - 启用代码建模
- `multi_modality`: True - 启用多模态融合
- `disable_gnn`: False - 是否禁用GNN
- `disable_src_code`: False - 是否禁用源代码分支

### 📁 数据配置
- `v_db`: "v21" - 数据库版本
- `dataset`: "programl" - 数据集类型
- `graph_type`: 程序图类型
- `batch_size`: 训练批次大小

### 🧠 模型配置
- `gnn_type`: "gat"/"gcn"/"transformer" - GNN类型
- `code_encoder`: "codet5" - 代码编码器
- `D`: 隐藏层维度
- `epoch_num`: 训练轮数 