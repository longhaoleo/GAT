# 项目框架
project/
> data/               # 存放数据集
> src/                # 代码
>
> > data.py   # 数据预处理
> > test.ipynb          # Jupyter 项目核心调用代码
> > test.py      # shap分析，消融实验
> > models/             # 存储训练好的模型
> > requirements.txt    # 依赖包
> > README.md           # 项目说明

# 环境准备
## R
提前下载R，并把R_HOME加入系统变量
在 R 中先安装 BiocManager:`install.packages("BiocManager")`
使用 BiocManager 安装 limma `BiocManager::install("limma")`

## python
`pip install -r requirements.txt`
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
清华镜像源：pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

