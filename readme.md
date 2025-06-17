# env setup

```
conda create -n coco python=3.9
```

```
conda activate coco
```

```
conda install pandas scikit-learn
```

## yaml

create new yaml

```
conda env export > coco_env.yaml
```

use it

```
conda env create -f coco_env.yaml
```



