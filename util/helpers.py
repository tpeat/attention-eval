from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt

def vis(image):
    image = image.permute(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def get_all_frames_labels(sample, gpu=0):
    ref_imgs = sample['ref_img']  # batch_size * 3 * h * w
    prev_imgs = sample['prev_img']
    curr_imgs = sample['curr_img']
    ref_labels = sample['ref_label']  # batch_size * 1 * h * w
    prev_labels = sample['prev_label']
    curr_labels = sample['curr_label']
    obj_nums = sample['meta']['obj_num']
    bs, _, h, w = curr_imgs[0].size()

    ref_imgs = ref_imgs.cuda(gpu, non_blocking=True)
    prev_imgs = prev_imgs.cuda(gpu, non_blocking=True)
    curr_imgs = [
        curr_img.cuda(gpu, non_blocking=True)
        for curr_img in curr_imgs
    ]
    ref_labels = ref_labels.cuda(gpu, non_blocking=True)
    prev_labels = prev_labels.cuda(gpu, non_blocking=True)
    curr_labels = [
        curr_label.cuda(gpu, non_blocking=True)
        for curr_label in curr_labels
    ]
    obj_nums = list(obj_nums)
    obj_nums = [int(obj_num) for obj_num in obj_nums]

    batch_size = ref_imgs.size(0)

    all_frames = torch.cat([ref_imgs, prev_imgs] + curr_imgs,
                           dim=0)
    all_labels = torch.cat([ref_labels, prev_labels] + curr_labels,
                                       dim=0)
    
    return all_frames, all_labels, batch_size, obj_nums


def load_network(net, pretrained_dir, gpu):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    if 'state_dict' in pretrained.keys():
        pretrained_dict = pretrained['state_dict']
    elif 'model' in pretrained.keys():
        pretrained_dict = pretrained['model']
    else:
        pretrained_dict = pretrained
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del (pretrained)
    return net.cuda(gpu), pretrained_dict_remove

@dataclass
class TrainingConfig:
    name: str = "AOT"
    root: str = '/gv1/projects/AI_Surrogate/dev/tristan/aot-benchmark/datasets/AOT'
    pretrain: str = "pretrain_model/mobilenet_v2-b0353104.pth"
    gpus: int = 1
    dist_start_gpu: int = 0
    datasets: str = 'AOT'
    dist_enable: bool = False
    data_workers: int = 2
    
TrainingConfig
    
def load_training_config(cfg):
    cfg.DIST_START_GPU = TrainingConfig.dist_start_gpu
    cfg.TRAIN_GPUS = TrainingConfig.gpus
    cfg.DATASETS = TrainingConfig.datasets
    cfg.DIR_AOT = TrainingConfig.root
    cfg.PRETRAIN_MODEL = TrainingConfig.pretrain
    cfg.DIST_ENABLE = TrainingConfig.dist_enable
    cfg.DATA_WORKERS = TraininConfig.data_workers
    return cfg
