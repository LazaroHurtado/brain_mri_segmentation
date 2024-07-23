import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import v2 as T

from dataloader import BrainMriDataLoader
from loss import Loss
from trainer import Trainer
from unet import UNet

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    image_transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = T.Compose([
        T.Grayscale(num_output_channels=1)
    ])
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90)
    ])

    brain_mri_dataloader = BrainMriDataLoader(transform, image_transform, mask_transform)

    model = UNet(in_channels=3, out_channels=1).to(rank)
    model = DDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_fn = Loss(criterion, gpu_id=rank)

    trainer = Trainer(model,
                      brain_mri_dataloader,
                      optimizer,
                      scheduler,
                      loss_fn,
                      gpu_id=rank)
    trainer.run()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()