import torch
import numpy as np
from tqdm import tqdm
from models.losses import prototype_loss
from cdfsl_dataset.meta_dataset_reader import MetaDatasetEpisodeReader
import argparse
from tabulate import tabulate
import tensorflow as tf
from models.sur import apply_selection, sur
from utils.model_utils import get_model, get_base_model


def get_multidomain_features(extractors, images, return_type='dict'):
    with torch.no_grad():
        all_features = dict()
        for name, extractor in extractors.items():
            all_features[name] = extractor.forward_features(images)
    if return_type == 'list':
        return list(all_features.values())
    else:
        return all_features


# def get_base_model(img_size, model_type, pretrained_ckpt='checkpoints/pretrained_ckpts/ViT-B_16.npz'):
#     config = CONFIGS[model_type]
#     model = VisionTransformer(config, img_size, zero_head=True, num_classes=1000)
#     model.load_from(np.load(pretrained_ckpt))
#     return model
#
#
# def get_model(model_dir):
#     config_file = os.path.join(model_dir,'model_config.json')
#     with open(config_file) as cfg:
#         config = json.load(cfg)
#     print('Getting model:')
#     print(config)
#
#     model = VisionTransformer(CONFIGS[config['model_type']], num_classes=config['num_classes'], img_size=config['img_size'], zero_head=True)
#     checkpoint_file = os.path.join(model_dir,'checkpoint.bin')
#     model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
#     return model


def eval_sur_metadataset(args):
    img_size = args.img_size
    trainsets = args.trainsets.split(' ')
    testsets = args.testsets.split(' ')

    config_file = f'cdfsl_dataset/configs/meta_dataset_{img_size}x{img_size}.gin'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(trainsets)
    extractors=dict()
    for trainset in trainsets:
        if trainset == 'imagenet':
            extractors[trainset] = get_base_model(img_size=img_size).to(device)
        else:
            extractors[trainset] = get_model(img_size, trainset).to(device)

    #
    #
    # extractors=dict()
    # extractors['imagenet'] = model0.to(device)
    # extractors['cifar'] = model1.to(device)
    # extractors['dtd'] = model1.to(device)
    #
    # trainsets = "omniglot".split(' ')
    # valsets = "omniglot".split(' ')
    # testsets = "omniglot mnist".split(' ')
    # print('train domains:', trainsets)
    # print('test domains:', testsets)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.disable_eager_execution()
    test_loader = MetaDatasetEpisodeReader('test', trainsets, testsets, testsets, config_file=config_file)

    N_TASKS = args.num_tasks
    accs_names = ['ViT']
    all_accs = dict()
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(f'Evaluating few shot classification on {dataset} dataset')
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
    parser.add_argument("--img_size", type=int, default=224,
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--trainsets", type=str, default="imagenet",
                        help="Use base ViT model")
    parser.add_argument("--testsets", type=str, default="mnist",
                        help="Evaluation domains")
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="Where to search for pretrained ViT models.")
    args = parser.parse_args()
    eval_sur_metadataset(args)


if __name__ == '__main__':
    main()