import argparse

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from utils.model_utils import *


def get_attention_mask(im, x, model):
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
    # result = (mask * im).astype("uint8")

    return mask


def visualize(model_dir, img_path, plot_layerwise_attentions=False, base=False):
    config_file = os.path.join(model_dir, 'model_config.json')
    with open(config_file) as cfg:
        config = json.load(cfg)
    print('Visualizing attention for model:')
    print(config)

    if base:
        model = VisionTransformer(CONFIGS[config['model_type']], num_classes=21843, zero_head=False,
                                  img_size=config['img_size'], vis=True)
        model.load_from(np.load('checkpoints/pretrained_ckpts/ViT-B_16.npz'))
    else:
        model = VisionTransformer(CONFIGS[config['model_type']], num_classes=config['num_classes'], zero_head=False,
                                  img_size=config['img_size'], vis=True)
        checkpoint_file = os.path.join(model_dir, 'checkpoint.bin')
        model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    im = Image.open(img_path)
    x = transform(im)

    mask = get_attention_mask(im, x, model)

    fig, ax1 = plt.subplots(ncols=1, figsize=(5, 5))

    ax1.set_title('Attention map')
    _ = ax1.imshow(im)
    _ = ax1.imshow(mask, cmap='jet', alpha=0.5)
    plt.show()

    return mask


def visualize_imgs(model_dir, img_dir, plot_layerwise_attentions=False, base=False):
    config_file = os.path.join(model_dir, 'model_config.json')
    with open(config_file) as cfg:
        config = json.load(cfg)
    print('Visualizing attention for model:')
    print(config)

    if base:
        model = VisionTransformer(CONFIGS[config['model_type']], num_classes=21843, zero_head=False,
                                  img_size=config['img_size'], vis=True)
        model.load_from(np.load('checkpoints/pretrained_ckpts/ViT-B_16.npz'))
    else:
        model = VisionTransformer(CONFIGS[config['model_type']], num_classes=config['num_classes'], zero_head=False,
                                  img_size=config['img_size'], vis=True)
        checkpoint_file = os.path.join(model_dir, 'checkpoint.bin')
        model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    for img_path in os.listdir(img_dir):
        if 'jpg' in img_path or 'png' in img_path or 'ppm' in img_path or 'JPG' in img_path:
            im = Image.open(os.path.join(img_dir,img_path))
            x = transform(im)
            mask = get_attention_mask(im, x, model)

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
            ax1.imshow(im)
            _ = ax2.imshow(im)
            _ = ax2.imshow(mask, cmap='jet', alpha=0.5)
            plt.show()
            fig.savefig(os.path.join('data/test_imgs', 'attn_images/attn_'+img_path.split('.')[0]+'.png'), bbox_inches='tight')

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="checkpoints/output_ckpts/cifar10_img224",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--test_img", type=str, default="data/test_imgs/traffic_sign2.png",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--img_dir", type=str, default="data/test_imgs",
                        help="Where to search for pretrained ViT models.")
    args = parser.parse_args()
    # visualize(args.model_config, args.test_img, plot_layerwise_attentions=False, base=True)
    visualize_imgs(args.model_config, args.img_dir, plot_layerwise_attentions=False, base=True)

    return


if __name__ == "__main__":
    main()
