import torch
from unet import UNet
from utils import load_data_set
from torchvision.transforms import transforms
from tqdm import tqdm
import torchvision

config = {
    "lr": 1e-3,
    "batch_size": 16,
    "image_dir": "CUB_200_2011/CUB_200_2011/images",
    "segmentation_dir": "CUB_200_2011/CUB_200_2011/segmentations",
    "image_paths": "CUB_200_2011/CUB_200_2011/images.txt",
    "epochs": 10,
    "checkpoint": "checkpoint/bird_segmentation_v1.pth",
    "optimiser": "checkpoint/bird_segmentation_v1_optim.pth",
    "continue_train": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"Training using {config['device']}")

transforms_image = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (1., 1., 1.))
])

transforms_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.,), (1.,))
])

train_dataset, val_dataset = load_data_set(
    config['image_paths'],
    config['image_dir'],
    config['segmentation_dir'],
    transforms=[transforms_image, transforms_mask],
    batch_size=config['batch_size']
)

print("loaded", len(train_dataset), "batches")

model = UNet(3).to(config['device'])
optimiser = torch.optim.Adam(params=model.parameters(), lr=config['lr'])

if config['continue_train']:
    state_dict = torch.load(config['checkpoint'])
    optimiser_state = torch.load(config['optimiser'])
    model.load_state_dict(state_dict)
    optimiser.load_state_dict(optimiser_state)

loss_fn = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

model.train()


def check_accuracy_and_save(model, optimiser, epoch):
    torch.save(model.state_dict(), config['checkpoint'])
    torch.save(optimiser.state_dict(), config['optimiser'])

    num_correct = 0
    num_pixel = 0
    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x, y in val_dataset:
            x = x.to(config['device'])
            y = y.to(config['device'])

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixel += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

            torchvision.utils.save_image(preds, f"test/pred/{epoch}.png")
            torchvision.utils.save_image(y, f"test/true/{epoch}.png")

    print(
        f"Dice Score = {dice_score/len(val_dataset)}"
    )
    model.train()


def train():
    step = 0
    for epoch in range(config['epochs']):
        loop = tqdm(train_dataset)
        for image, seg in loop:
            image = image.to(config['device'])
            seg = seg.float().to(config['device'])

            with torch.cuda.amp.autocast():
                pred = model(image)
                loss = loss_fn(pred, seg)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            loop.set_postfix(loss=loss.item())
            step += 1
        check_accuracy_and_save(model, optimiser, epoch)


if __name__ == "__main__":
    train()
