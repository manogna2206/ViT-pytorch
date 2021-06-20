import argparse
import numpy as np
import tensorflow as tf
import torch
from tabulate import tabulate
from tqdm import tqdm
from cdfsl_dataset.meta_dataset_reader import MetaDatasetEpisodeReader
from models.losses import prototype_loss
from utils.model_utils import get_model, get_base_model


def eval_metadataset(args):
    img_size = args.img_size
    trainset = args.trainset
    testsets = args.testsets
    if trainset == 'imagenet':
        model = get_base_model(img_size=img_size, from_timm=True)
    else:
        model = get_model(img_size, trainset, from_timm=True)

    config_file = f'cdfsl_dataset/configs/meta_dataset_{img_size}x{img_size}.gin'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    testsets = testsets.split(' ')
    print('Evaluating on test domains:', testsets)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    test_loader = MetaDatasetEpisodeReader('test', testsets, testsets, testsets, config_file=config_file)

    N_TASKS = args.num_tasks
    accs_names = ['ViT']
    var_accs = dict()
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}
            for i in tqdm(range(N_TASKS)):
                with torch.no_grad():
                    sample = test_loader.get_test_task(session, dataset)
                    context_features = model.forward_features(sample['context_images'])
                    target_features = model.forward_features(sample['target_images'])
                    context_labels = sample['context_labels'].to(device)
                    target_labels = sample['target_labels'].to(device)
                    _, stats_dict_ViT, _ = prototype_loss(context_features, context_labels,
                                                          target_features, target_labels)
                    vit_acc = stats_dict_ViT['acc']
                    # print(f'Accuracy of ViT test task {i}:{vit_acc}')

                var_accs[dataset]['ViT'].append(stats_dict_ViT['acc'])

    # Print SUR results table
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n\n")
    return


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--img_size", type=int, default=84,
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--trainset", type=str, default="imagenet",
                        help="Use base ViT model")
    parser.add_argument("--testsets", type=str, default="mnist",
                        help="Evaluation domains")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Where to search for pretrained ViT models.")
    args = parser.parse_args()
    eval_metadataset(args)


if __name__ == '__main__':
    main()
