# CrossHomo

## 1.Requirements
- Python >= 3.7
- Pytorch >= 0.4.1
- opencv-python
- korina
- matplotlib

## 2.Dataset
DPDN and RGB-NIR Scene Dataset.

## 3.Training

### Train for 8x resolution gap
Run the following command.
```
python train_SR_Homo_level3.py --name Level3 --dataset DPDN --batch_size 8 --downsample 8 --Level 3 --load_latest 0
```

### Train for 4x resolution gap
Run the following command.
```
python train_SR_Homo_level2.py --name Level2 --dataset DPDN --batch_size 8 --downsample 4 --Level 2 --load_latest 0
```

## 4.Checkpoints
链接：https://pan.baidu.com/s/1_z6YI_BkEfyd7jXqj9QWmA?pwd=BuAa 
提取码：BuAa 
