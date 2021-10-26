import torch
import os, time, sys, cv2, tqdm
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import PictorV3Dataset, Transformer, collate_source_fn, collate_valid_fn, i2c
import config as cfg
from model import Model
from loss import Criterion
from utils import unnormalize, Inference, GtTransform, Evaluation

use_gpu = torch.cuda.is_available()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, torch.nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, torch.nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, torch.nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, torch.nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def visualize(image, fname):
    fig, axs = plt.subplots()

    axs.imshow(image)
    axs.axis('off')
    fig.tight_layout()
    # plt.show()
    plt.savefig(fname + '.svg', format='svg', dpi=1000)
    plt.close('all')

def train():
    source_train_dataset = PictorV3Dataset(image_dir='./dataset/source/images/', label_dir='./dataset/source/labels/', set_name='source', scale='large', transform=Transformer())
    valid_dataset = PictorV3Dataset(image_dir='./dataset/valid/images/', label_dir='./dataset/valid/labels/', set_name='valid', scale='large', transform=Transformer())

    source_train_loader = DataLoader(source_train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.n_jobs, collate_fn=collate_source_fn, pin_memory=use_gpu)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=cfg.n_jobs, collate_fn=collate_valid_fn, pin_memory=use_gpu)

    model = torch.nn.DataParallel(Model())
    criterion = Criterion()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=cfg.lr)

    inference = Inference()
    gttrans = GtTransform()
    eval = Evaluation()

    if use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        criterion.cuda()

    print('Number of parameters: ', count_parameters(model))

    curr_epoch = 0
    if os.path.exists(cfg.logdir + 'training_state.pth'):
        if use_gpu:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load(cfg.logdir + 'training_state.pth', map_location=map_location)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        curr_epoch = ckpt['global_step']
        print('Restore model')

    
    for epoch in range(curr_epoch+1, cfg.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (weak_source, strong_source, gt_clses_batch, gt_bboxes_batch, trans_gt_bboxes_batch) in enumerate(source_train_loader):
            start = time.time()

            if use_gpu:
                weak_source, strong_source = weak_source.cuda(), strong_source.cuda()
                gt_clses_batch = [gt_clses.cuda() for gt_clses in gt_clses_batch]
                gt_bboxes_batch = [gt_bboxes.cuda() for gt_bboxes in gt_bboxes_batch]
                trans_gt_bboxes_batch = [trans_gt_bboxes.cuda() for trans_gt_bboxes in trans_gt_bboxes_batch]

            weak_source_cls_logit_batch, weak_source_reg_logit_batch = model(weak_source)
            strong_source_cls_logit_batch, strong_source_reg_logit_batch = model(strong_source)

            weak_gt_qfl_batch, weak_gt_dfl_batch, anchor_points, fpn_strides = gttrans(gt_clses_batch, gt_bboxes_batch, weak_source_reg_logit_batch.detach())
            strong_gt_qfl_batch, strong_gt_dfl_batch, _, _ = gttrans(gt_clses_batch, trans_gt_bboxes_batch, strong_source_reg_logit_batch.detach())

            ## perform losses
            weak_source_qfl_loss, weak_source_dfl_loss, weak_source_giou_loss = criterion(weak_gt_qfl_batch, weak_gt_dfl_batch, anchor_points, fpn_strides, weak_source_cls_logit_batch, weak_source_reg_logit_batch)
            strong_source_qfl_loss, strong_source_dfl_loss, strong_source_giou_loss = criterion(strong_gt_qfl_batch, strong_gt_dfl_batch, anchor_points, fpn_strides, strong_source_cls_logit_batch, strong_source_reg_logit_batch)

            weak_source_loss = weak_source_qfl_loss + weak_source_dfl_loss + 2 * weak_source_giou_loss
            strong_source_loss = strong_source_qfl_loss + strong_source_dfl_loss + 2 * strong_source_giou_loss
            loss = weak_source_loss + strong_source_loss

            loss.backward()

            if (step+1) % cfg.update == 0 or (step+1) == len(source_train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            end = time.time()
            sys.stdout.write('\rEpoch: %03d, Step: %04d/%d, Loss: %.9f, Time training: %.2f secs' % (epoch, step+1, len(source_train_loader), loss.item(), end-start))


        if epoch % cfg.save == 0 or epoch == cfg.epochs:
            ckpt_path = cfg.logdir + 'training_state.pth'
            torch.save({'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': epoch}, ckpt_path)

            ckpt_path = cfg.logdir + 'parameters.pth'
            torch.save({'state_dict': model.state_dict()}, ckpt_path)


        if epoch % cfg.evaluate == 0 or epoch == cfg.epochs:
            model.eval()
            for image, gt_clses_batch, gt_bboxes_batch in tqdm.tqdm(valid_loader):
                if use_gpu:
                    image = image.cuda()
                    gt_clses_batch = [gt_clses.cuda() for gt_clses in gt_clses_batch]
                    gt_bboxes_batch = [gt_bboxes.cuda() for gt_bboxes in gt_bboxes_batch]

                with torch.no_grad():
                    pred_cls_batch, pred_reg_batch = model(image)
                    pred_cls_batch = pred_cls_batch.sigmoid()
                    pred_reg_batch = pred_reg_batch.softmax(dim=2)

                    pred_clses_batch, pred_bboxes_batch, pred_quality_scores_batch = inference(pred_cls_batch, pred_reg_batch, threshold=0.5)

                    eval.append(gt_clses_batch, gt_bboxes_batch, pred_clses_batch, pred_bboxes_batch, pred_quality_scores_batch)
                
            mAP = eval.eval()
            print('mAP:', mAP)

            image = unnormalize(image, mean=cfg.mean, std=cfg.std)
            image = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8').copy()

            for (cls_id, (ymin, xmin, ymax, xmax), score) in zip(pred_clses_batch[0].cpu().numpy().astype('int'), pred_bboxes_batch[0].cpu().numpy().astype('int'), pred_quality_scores_batch[0].cpu().numpy()):
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, i2c[cls_id] + ' %.2f' % (score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            for (cls_id, (ymin, xmin, ymax, xmax)) in zip(gt_clses_batch[0].cpu().numpy().astype('int'), gt_bboxes_batch[0].cpu().numpy().astype('int')):
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.putText(image, i2c[cls_id], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            visualize(image, 'valid')











if __name__ == '__main__':
    print('Use GPU: ', use_gpu)
    if use_gpu:
        print('Device name: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    print('Object Detection\n')
    train()
