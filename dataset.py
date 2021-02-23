from torch.utils.data import Dataset
from PIL import Image
import os


class BirdDataset(Dataset):
    def __getitem__(self, index):
        image_name = ".".join(self.images_paths[index].split('.')[:-1])

        image = Image.open(os.path.join(self.image_dir, f"{image_name}.jpg")).convert("RGB")
        seg = Image.open(os.path.join(self.segmentation_dir, f"{image_name}.png")).convert("L")

        image = self.transform_image(image)
        seg = self.transform_mask(seg)

        return image, seg

    def __init__(self, image_paths, image_dir, segmentation_dir, transform_image, transform_mask):
        super(BirdDataset, self).__init__()
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        with open(image_paths, 'r') as f:
            self.images_paths = [line.split(" ")[-1] for line in f.readlines()]

    def __len__(self):
        return len(self.images_paths)


def test():
    image_dir = "CUB_200_2011/CUB_200_2011/images"
    segmentation_dir = "CUB_200_2011/CUB_200_2011/segmentations"
    image_paths = "CUB_200_2011/CUB_200_2011/images.txt"

    dataset = BirdDataset(image_paths, image_dir, segmentation_dir)

    image, seg = dataset[0]
    # image.show()
    seg.show()


if __name__ == "__main__":
    test()
