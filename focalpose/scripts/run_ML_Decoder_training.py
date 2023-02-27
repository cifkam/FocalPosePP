import cv2
import torch
import argparse
import torchvision
import numpy as np
from pathlib import Path
import pickle
import os

from PIL import Image
from tqdm import tqdm
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

from randaugment import RandAugment

import torchvision.transforms as transforms

from sklearn.preprocessing import LabelEncoder

from focalpose.config import EXP_DIR, LOCAL_DATA_DIR, FEATURES_DIR, CLASSIFIERS_ML_DECODER_DIR
from focalpose.utils.resources import assign_gpu
from focalpose.utils.logging import get_logger
from focalpose.rendering.bullet_batch_renderer import BulletBatchRenderer
from focalpose.rendering.bullet_scene_renderer import BulletSceneRenderer
from focalpose.datasets.datasets_cfg import make_urdf_dataset, make_scene_dataset
from focalpose.datasets.pose_dataset import PoseDataset
from focalpose.lib3d.focalpose_ops import TCO_init_from_boxes_zup_autodepth
from focalpose.training.pose_models_cfg import create_model_pose
from focalpose.lib3d.rigid_mesh_database import MeshDataBase
from focalpose.models.model_classifier import ModelClassifier, ClassifierSubnet
from focalpose.datasets.ml_decoder_detection_dataset import MLDecoderDetectionDataset

from ML_Decoder.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, add_weight_decay
from ML_Decoder.models import create_model
from ML_Decoder.loss_functions.losses import AsymmetricLoss

cudnn.benchmark = False

logger = get_logger(__name__)

def validate_multi(val_loader, model, ema_model):
    logger.info("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    targets = torch.cat(targets).numpy()
    preds_ema = torch.cat(preds_ema).numpy()
    preds_regular = torch.cat(preds_regular).numpy()

    mAP_score_regular = mAP(targets, preds_regular)
    mAP_score_ema     = mAP(targets, preds_ema)
    logger.info(f"mAP score regular {mAP_score_regular:.2f}, mAP score EMA {mAP_score_ema:.2f}")

    targets = np.argmax(targets, axis=1)
    top_5_regular = []
    top_5_ema = []
    top_1_regular = []
    top_1_ema = []
    for idx in range(len(targets)):

        top_5 = np.argsort(preds_regular[idx])[::-1][:5]
        if targets[idx] in top_5: top_5_regular.append(1)
        else: top_5_regular.append(0)
        if targets[idx] == top_5[0]: top_1_regular.append(1)
        else: top_1_regular.append(0)

        top_5 = np.argsort(preds_regular[idx])[::-1][:5]
        if targets[idx] in top_5: top_5_ema.append(1)
        else: top_5_ema.append(0)
        if targets[idx] == top_5[0]: top_1_ema.append(1)
        else: top_1_ema.append(0)

    val_mAP = max(mAP_score_regular, mAP_score_ema)
    val_acc1 = max(np.mean(top_1_regular), np.mean(top_1_ema))
    val_acc5 = max(np.mean(top_5_regular), np.mean(top_5_ema))

    logger.info(f'Top-1 Acc: {val_acc1*100:.2f}')
    logger.info(f'Top-5 Acc: {val_acc5*100:.2f}')

    return val_mAP, val_acc1, val_acc5

def run_training(model, train_loader, val_loader, lr):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=0.2)

    best_acc1 = 0
    best_acc5 = 0
    trainInfoList = []
    scaler = GradScaler()

    for epoch in range(Epochs):
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                logger.info(f"Epoch [{epoch}/{Epochs}], Step [{str(i).zfill(3)}/{str(steps_per_epoch).zfill(3)}], LR {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.1f}")

        model.eval()
        mAP_score, acc1, acc5 = validate_multi(val_loader, model, ema)
        model.train()

        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            path = CLASSIFIERS_ML_DECODER_DIR / f"model-{cfg.config}.ckpt"
            try:
                torch.save(model.state_dict(), path)
            except Exception as e:
                logger.info(f"failed to save model {path}: {e}")

    logger.info("="*42)
    logger.info(f"config: {cfg.config}")
    logger.info(f"Best: Top-1 Acc: {(best_acc1*100):.2f}, Top-5 Acc: {(best_acc5*100):.2f}")
    logger.info("="*42 + "\n")




def main(cfg):

    def make_datasets(dataset_names):
        datasets = []
        all_labels = set()

        for (ds_name, n_repeat, n_frames) in dataset_names:
            ds_train = make_scene_dataset(ds_name, n_frames)
            for _ in range(n_repeat):
                datasets.append(ds_train)

            logger.info(f'Loaded {ds_name} with {len(ds_train) * n_repeat} images.')
            all_labels = all_labels.union(set(ds_train.all_labels))

        return ConcatDataset(datasets), all_labels

    scene_ds_train, train_labels = make_datasets(cfg.train_ds_names)
    scene_ds_val, val_labels = make_datasets(cfg.test_ds_names)
    all_labels = train_labels.union(val_labels)

    labels = sorted(list(all_labels))
    labels_to_id = {}
    id_to_labels = {}
    for idx, label in enumerate(labels):
        labels_to_id[label] = idx
        id_to_labels[idx] = label

    train_dataset = MLDecoderDetectionDataset(
        scene_ds_train,
        labels_to_id, 
        transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            # normalize,
    ]))
    val_dataset = MLDecoderDetectionDataset(
        scene_ds_val,
        labels_to_id,
        transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            # normalize, # no need, toTensor does normalization
    ]))

    model_cfg = argparse.ArgumentParser('').parse_args([])
    model_cfg.model_name = 'tresnet_l'
    model_cfg.model_path = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth'
    model_cfg.num_classes = len(labels)
    model_cfg.num_of_groups = -1
    model_cfg.use_ml_decoder = True
    model_cfg.decoder_embedding = 768
    model_cfg.zsl = False

    logger.info(f"creating model {model_cfg.model_name}...")
    model = create_model(model_cfg).cuda()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=False)

    # Actuall Training
    run_training(model, train_loader, val_loader, cfg.lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classifier training')
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--batch-size', default=56, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=1e-4, type=float)

    assign_gpu()
    torch.manual_seed(42)
    np.random.seed(42)

    args = parser.parse_args()
    cfg = argparse.ArgumentParser('').parse_args([])
    cfg.config = args.config
    cfg.workers = args.workers
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.image_size = 384


    if 'pix3d-sofa' in args.config:
        cfg.urdf_ds_name = 'pix3d-sofa'
        cfg.n_symmetries_batch = 1

        if 'real+synth1k' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, 1000), ('pix3d-sofa.train', 1, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'F50p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, 551), ('pix3d-sofa.train', 1, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-sofa.train', 1, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]

    elif 'pix3d-bed' in args.config:
        cfg.urdf_ds_name = 'pix3d-bed'
        cfg.n_symmetries_batch = 1

        if 'real+synth1k' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1, 1000), ('pix3d-bed.train', 1, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]
        elif 'F50p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1, 203), ('pix3d-bed.train', 1, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-bed.train', 1, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]

    elif 'pix3d-table' in args.config:
        cfg.urdf_ds_name = 'pix3d-table'
        cfg.n_symmetries_batch = 1

        if 'real+synth1k' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1, 1000), ('pix3d-table.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]
        elif 'F50p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1, 386), ('pix3d-table.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-table.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]

    elif 'pix3d-chair' in args.config:
        cfg.urdf_ds_name = 'pix3d-chair'
        cfg.n_symmetries_batch = 1

        if 'real+synth1k' in args.config:
            cfg.train_ds_names = [(f'synthetic.pix3d-chair-1-1M.train', 1, 1000)] +\
                                 [('pix3d-chair.train', 1, None)]
            cfg.test_ds_names =  [('pix3d-chair.test', 1, None)]
        elif 'F50p' in args.config:
            cfg.train_ds_names = [(f'synthetic.pix3d-chair-1-1M.train', 1, 1506)] +\
                                 [('pix3d-chair.train', 1, None)]
            cfg.test_ds_names =  [('pix3d-chair.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-chair.train', 1, None)]
            cfg.test_ds_names =  [('pix3d-chair.test', 1, None)]

    elif 'compcars3d' in args.config:
        cfg.urdf_ds_name = 'compcars3d'
        cfg.n_symmetries_batch = 1

        if 'real+synth1k' in args.config:
            cfg.train_ds_names = [(f'synthetic.compcars3d-1-1M.train', 1, 1000)] +\
                                 [('compcars3d.train', 1, None)]
            cfg.test_ds_names =  [('compcars3d.test', 1, None)]
        elif 'F50p' in args.config:
            cfg.train_ds_names = [(f'synthetic.compcars3d-1-1M.train', 1, 3798)] +\
                                 [('compcars3d.train', 1, None)]
            cfg.test_ds_names =  [('compcars3d.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('compcars3d.train', 1, None)]
            cfg.test_ds_names =  [('compcars3d.test', 1, None)]

    elif 'stanfordcars3d' in args.config:
        cfg.urdf_ds_name = 'stanfordcars3d'
        cfg.n_symmetries_batch = 1

        if 'real+synth1k' in args.config:
            cfg.train_ds_names = [(f'synthetic.stanfordcars3d-1-1M.train', 1, 1000)] +\
                                 [('stanfordcars3d.train', 1, None)]
            cfg.test_ds_names =  [('stanfordcars3d.test', 1, None)]
        elif 'F50p' in args.config:
            cfg.train_ds_names = [(f'synthetic.stanfordcars3d-{1}-1M.train', 1, 8144)] +\
                                 [('stanfordcars3d.train', 1, None)]
            cfg.test_ds_names =  [('stanfordcars3d.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('stanfordcars3d.train', 1, None)]
            cfg.test_ds_names =  [('stanfordcars3d.test', 1, None)]


    main(cfg=cfg)
