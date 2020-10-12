from tqdm.auto import tqdm
from experiment_utils.dataset import get_dataloader_incr

# Test incremental dataloading
if __name__ == '__main__':
    train_loaders, val_loaders = get_dataloader_incr(train=True, val_size=200)
    for train_loader, val_loader in zip(train_loaders, val_loaders):
        for _ in tqdm(train_loader):
            pass
        for _ in tqdm(val_loader):
            pass
    test_loaders = get_dataloader_incr(train=False)
    for loader in test_loaders:
        for _ in tqdm(loader):
            pass
    pass
