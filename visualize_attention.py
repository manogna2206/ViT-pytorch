import numpy as np
import torch
import cv2
import os, json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS


def visualize(model_dir, plot_layerwise_attentions=False):

    config_file = os.path.join(model_dir,'model_config.json')
    with open(config_file) as cfg:
        config = json.load(cfg)
    print('Visualizing attention for model:')
    print(config)

    model = VisionTransformer(CONFIGS[config['model_type']], num_classes=config['num_classes'], zero_head=False, img_size=config['img_size'], vis=True)
    checkpoint_file = os.path.join(model_dir,'checkpoint.bin')
    model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    im = Image.open("data/test_imgs/dog.jpg")
    x = transform(im)

    logits, att_mat = model(x.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)
    plt.show()

    if plot_layerwise_attentions:
        for i, v in enumerate(joint_attentions):
            # Attention from the output token to the input space.
            mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
            result = (mask * im).astype("uint8")

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
            ax1.set_title('Original')
            ax2.set_title('Attention Map_%d Layer' % (i + 1))
            _ = ax1.imshow(im)
            _ = ax2.imshow(result)
            plt.show()

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="output/cifar10_img84_ncls47",
                        help="Where to search for pretrained ViT models.")
    args = parser.parse_args()
    visualize(args.model_config, plot_layerwise_attentions=True)

    return


if __name__ == "__main__":
    main()