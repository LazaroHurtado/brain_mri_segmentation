
UNET model applied to the Kaggle [Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data) dataset

## Usage

```zsh
>> python3 -m venv venv
>> pip3 install -r requirements.txt
>> torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:12345 main.py
```

If you want to run on multiple GPUs, you can increase the `--nproc_per_node` argument.

Loss function is a weighted linear combination of [dice loss](https://arxiv.org/pdf/1606.04797) and [focal loss](https://arxiv.org/pdf/1708.02002). Focal loss is useful for imbalanced datasets but I also use the inverse class frequency when calculating the BCE.