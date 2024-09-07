from base_unet.model.UNet3Plus import UNet3Plus as a
# model = Unet(num_classes=4)
model = a()
# model = torch.load(r"D:\AILAB\校园算法\挑战赛\Rein\checkpoints\dinov2_rein_and_head.pth")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f'Total number of trainable parameters: {total_params}')
