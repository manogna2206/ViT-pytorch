import torch
import gin
import numpy as np
from tqdm import tqdm
from models.losses import prototype_loss
from cdfsl_dataset.meta_dataset_reader import MetaDatasetEpisodeReader
from train import get_model
import argparse
from tabulate import tabulate
import tensorflow as tf
from models.sur import apply_selection, sur


def get_multidomain_features(extractors, images, return_type='dict'):
    with torch.no_grad():
        all_features = dict()
        for name, extractor in extractors.items():
            all_features[name] = extractor.forward_features(images)
    if return_type == 'list':
        return list(all_features.values())
    else:
        return all_features


@gin.configurable()
def eval_metadataset(args, img_size):
    args, model = get_model(args, training=False)
    config_file = f'cdfsl_dataset/configs/meta_dataset_{img_size}x{img_size}.gin'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    extractors=dict()
    extractors['imagenet'] = model
    extractors['cifar'] = model

    trainsets = "omniglot".split(' ')
    valsets = "omniglot".split(' ')
    testsets = "mnist".split(' ')
    print('train domains:', trainsets)
    print('test domains:', testsets)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, config_file=config_file)
    test_loader = MetaDatasetEpisodeReader('test', trainsets, valsets, testsets, config_file=config_file)

    N_TASKS = 10
    accs_names = ['ViT']
    all_accs = dict()
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(f'Evaluation few shot classification on {dataset}')
            all_accs[dataset] = {name: [] for name in accs_names}
            for i in tqdm(range(N_TASKS)):
                sample = test_loader.get_test_task(session, dataset)
                context_features_dict = get_multidomain_features(extractors, sample['context_images'])
                target_features_dict = get_multidomain_features(extractors, sample['target_images'])
                context_labels = sample['context_labels'].to(device)
                target_labels = sample['target_labels'].to(device)

                selection_params = sur(context_features_dict, context_labels, max_iter=40)
                selected_context = apply_selection(context_features_dict, selection_params)
                selected_target = apply_selection(target_features_dict, selection_params)

                final_acc = prototype_loss(selected_context, context_labels,
                                               selected_target, target_labels)[1]['acc']
                all_accs[dataset]['ViT'].append(final_acc)


    # Print ViT-SUR results table
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(all_accs[dataset_name][model_name]) * 100
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
    parser.add_argument("--model_config", type=str, default="vit_configs/cifar_84.gin",
                        help="Where to search for pretrained ViT models.")
    args = parser.parse_args()
    gin.parse_config_file(args.model_config)
    eval_metadataset(args)


if __name__ == '__main__':
    main()