import torch
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        self.transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace(".jpg", ".png")) for image_id in self.ids]

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=image, mask=mask)
        return transformed['image'], transformed['mask'][:, :, 0]

    def __len__(self):
        return len(self.ids)


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # Adjust for DataParallel by removing 'module.' from key names
    state_dict = checkpoint
    new_state_dict = {}

    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()


def inference_and_save(model, data_loader, device, save_dir):
    model.eval()  # Set model to evaluation mode
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, _) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
            images = images.to(device)
            predictions = model(images)
            predictions = predictions.argmax(dim=1).cpu().numpy()

            for idx, pred in enumerate(predictions):
                result_path = os.path.join(save_dir, f"{i * len(images) + idx + 1:06d}.png")
                cv2.imwrite(result_path, pred)


def main_inference():
    x_test_dir = "../NEU_Seg-main/NEU_Seg-main/images/test"
    y_test_dir = "../NEU_Seg-main/NEU_Seg-main/annotations/test"
    test_dataset = SegDataset(x_test_dir, y_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Unet(num_classes=4).to(device)
    model = torch.load("../base_unet/checkpoints/CGNet_train1/CGNet_20_train1.pt")

    # checkpoint_path = 'checkpoints/Unet_5.pth'  # Adjust the path to your model checkpoint
    # load_model_checkpoint(model, checkpoint_path)
    # model = torch.load("checkpoints/Unet_5.pth").to(device)

    save_dir = "../base_unet/inference_results"
    inference_and_save(model, test_loader, device, save_dir)
    print(f"Results saved in {save_dir}")


if __name__ == "__main__":
    main_inference()
