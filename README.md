# AnyLoc地图瓦片检索系统

基于DINOv2和VLAD的视觉地理定位系统，用于无人机图像与地图瓦片的匹配检索。

## 项目特性

- 🚀 **高精度视觉定位**：使用DINOv2 ViT-G/14深度特征提取
- 🗺️ **地理信息集成**：融合视觉特征与地理坐标元数据
- ⚡ **GPU加速检索**：支持FAISS GPU索引，毫秒级检索
- 📊 **性能分析**：详细的执行时间分解和性能监控
- 🔧 **模块化设计**：清晰的工程结构，易于扩展和维护

## 工程结构

```
aerial_map_retrieval/
│
├── config/                          # 配置文件目录
│   └── config.yaml                  # 主配置文件（模型、路径、参数）
│
├── models/                          # 模型文件目录
│   └── aerial/
│       └── c_centers.pt            # VLAD聚类中心（32个cluster）
│
├── src/                            # 源代码目录
│   ├── __init__.py
│   │
│   ├── core/                       # 核心模块
│   │   ├── __init__.py
│   │   ├── models.py              # 数据模型定义
│   │   │   ├── TileMetadata      # 瓦片元数据
│   │   │   ├── RetrievalResult   # 检索结果
│   │   │   └── DescriptorData    # 描述符数据
│   │   └── schemas.py             # 数据映射和接口定义
│   │
│   ├── feature/                    # 特征处理模块
│   │   ├── __init__.py
│   │   ├── preprocessor.py       # 图像预处理
│   │   │   └── ImagePreprocessor # 统一预处理接口
│   │   ├── extractor.py          # 特征提取
│   │   │   └── DinoV2Extractor  # DINOv2特征提取器
│   │   └── aggregator.py         # 特征聚合
│   │       └── VLADAggregator   # VLAD聚合器
│   │
│   ├── retrieval/                  # 检索模块
│   │   ├── __init__.py
│   │   ├── engine.py              # 检索引擎
│   │   │   └── RetrievalEngine   # 主检索类
│   │   └── indexer.py            # 索引管理
│   │       └── FAISSIndexer      # FAISS索引管理器
│   │
│   ├── data/                       # 数据管理模块
│   │   ├── __init__.py
│   │   └── manager.py             # 元数据管理
│   │       └── MetadataManager   # 元数据管理器
│   │
│   └── utils/                      # 工具模块
│       ├── __init__.py
│       └── logger.py              # 日志配置
│
├── scripts/                        # 执行脚本
│   ├── extract_features.py       # 批量特征提取脚本
│   └── run_retrieval.py          # 检索系统运行脚本
│
├── data/                          # 数据目录（示例结构）
│   ├── source_tiles/              # 数据库瓦片
│   │   ├── img/                  # 瓦片图像
│   │   └── data.csv              # 瓦片元数据
│   ├── target_drone_img/         # 查询图像
│   ├── descriptors/              # 提取的描述符
│   └── results/                  # 检索结果
│
└── README.md                      # 项目文档
```

## 核心模块说明

### 1. 特征提取流程 (feature/)
```
图像 → 预处理 → DINOv2提取 → VLAD聚合 → 描述符
```
- **预处理**：统一resize到14的倍数（ViT patch大小）
- **特征提取**：使用DINOv2第31层的Value分支
- **VLAD聚合**：32个聚类中心，生成49152维描述符

### 2. 检索流程 (retrieval/)
```
查询图像 → 提取描述符 → FAISS检索 → 融合元数据 → 返回结果
```
- **索引构建**：L2归一化 + 内积 = 余弦相似度
- **GPU加速**：支持FAISS GPU索引
- **元数据融合**：匹配地理坐标信息

### 3. 数据管理 (data/)
- **元数据加载**：从CSV读取瓦片地理信息
- **坐标映射**：文件名与地理坐标关联

## 安装

### 环境要求
- Python >= 3.11
- CUDA >= 11.6 (GPU加速)

### 依赖安装
```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# FAISS (GPU版本)
pip install faiss-gpu

# 其他依赖
pip install pandas pillow pyyaml tqdm numpy
```

## 使用方法

### 1. 配置文件
编辑 `config/config.yaml` 设置数据路径和参数：
```yaml
data:
  database_dir: "/path/to/database/images"
  query_dir: "/path/to/query/images"
  metadata_file: "/path/to/metadata.csv"
```

### 2. 提取特征
```bash
# 提取数据库和查询图像特征
python scripts/extract_features.py
```

### 3. 运行检索
```bash
# 执行检索并输出结果
python scripts/run_retrieval.py
```

## 性能指标

典型性能（ViT-G/14, 896×896, RTX 3090）：
- 预处理：~15ms
- DINOv2推理：~450ms
- VLAD聚合：~50ms
- FAISS检索（6800个数据库）：~5ms
- **总计：~520ms/查询**

## API使用示例

```python
from src.retrieval.engine import RetrievalEngine
import yaml

# 加载配置
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化引擎
engine = RetrievalEngine(config)

# 加载数据库
engine.load_database(
    descriptor_file="data/descriptors/database_descriptors.pt",
    metadata_file="data/source_tiles/data.csv"
)

# 执行查询
results = engine.query("query_image.jpg", top_k=10)

# 输出结果
for r in results:
    print(f"Rank {r.rank}: {r.tile_metadata.filename}")
    print(f"  Score: {r.similarity_score:.4f}")
    print(f"  Location: ({r.tile_metadata.center_lat:.6f}, {r.tile_metadata.center_long:.6f})")
```

## 元数据格式

CSV文件格式要求：
```csv
Filename,Top_left_lat,Top_left_long,Bottom_right_lat,Bottom_right_long,zoom_level,spatial_resolution
1.jpg,30.286938,103.807067,30.284973,103.810101,18,0.405
```

## 技术细节

- **特征维度**：49152 (32 clusters × 1536 dim)
- **相似度度量**：余弦相似度（归一化内积）
- **索引类型**：FAISS IndexFlatIP（精确搜索）
- **预处理策略**：
  - 数据库图像：向上取整到14的倍数
  - 查询图像：保留原始比例，向下取整

## 参考文献

基于 [AnyLoc](https://github.com/AnyLoc/AnyLoc) 项目：
```
@InProceedings{Keetha_2023_CVPR,
    author    = {Keetha, Nikhil and Mishra, Avneesh and Karhade, Jay and Jatavallabhula, Krishna Murthy and Scherer, Sebastian and Krishna, Madhava and Garg, Sourav},
    title     = {AnyLoc: Towards Universal Visual Place Recognition},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
```

## License

MIT License