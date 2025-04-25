## 环境配置
 创建虚拟环境
```shell
conda create -n Annotated_Transformer python=3.8.19
```

 激活虚拟环境
```shell
conda activate Annotated_Transformer
```
 导入整个requirement文档
```shell
pip install -r "requirements.txt"

```

## multi30k数据集

```angular2html

URL = {
    "train": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "valid": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
}
```

## 训练
```shell
PYTHONPATH=$PYTHONPATH:. python scripts/examples/translation_task.py
```

## cuda 版本
```shell
pip uninstall torch
pip install torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu118
```

参考：
* https://nlp.seas.harvard.edu/annotated-transformer/
* https://zhuanlan.zhihu.com/p/105493618
