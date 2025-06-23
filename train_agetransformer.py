import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torch.nn.functional import mse_loss,l1_loss
from torch.nn.modules.distance import PairwiseDistance
from mean_variance_loss import MeanVarianceLoss
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from loss_function import compute_id_loss
from perceptual_loss import PerceptualLoss

from AdaFace.head import AdaFace
try:
    import wandb

except ImportError:
    wandb = None

from model_D_add_fc import Generator, Discriminator
from model_irse import IR_50
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def one_hot(dim, tmp):
    ones = torch.eye(dim)
    return ones.index_select(0, tmp)


def make_label(dim, batch):
    tmp = torch.LongTensor(np.random.randint(dim, size=batch))
    code = one_hot(dim, tmp)
    label = torch.LongTensor(tmp)
    return code, label

def make_specific_label(label, dim, batch):
    tmp = torch.LongTensor([label] * batch)
    code = one_hot(dim, tmp)
    return code

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 1, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def train(args, train_loader, val_loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    train_loader = sample_data(train_loader)
    val_loader = sample_data(val_loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    criterion1 = MeanVarianceLoss(0.1, 0.02, 0, 9).cuda()
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator
        
    accum = 0.5 ** (32 / (10 * 1000))

    # sample_z = torch.randn(args.n_sample, args.latent, device=device)
    z_repeat = args.n_sample // args.batch
    sample_z = []
    for _ in range(z_repeat):
        sample, _ = next(val_loader)
        sample_z.append(sample)
    sample_z = torch.stack(sample_z).view(args.n_sample, 3, args.size, args.size)
    utils.save_image(
                     sample_z,
                     f'sample/real.png',
                     nrow=int(args.n_sample ** 0.5),
                     normalize=True,
                     value_range=(-1, 1),
                    )
    sample_z = sample_z.to(device)
    weight = [1,1,1,1,1,1,1,1,1,1]
    ageclass_weights = torch.FloatTensor(weight).to(device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break
    
        train_img, train_atr = next(train_loader)
        train_atr_one = torch.from_numpy(np.eye(10)[train_atr]).type(torch.FloatTensor)
        
        train_img = train_img.to(device)
        train_atr = train_atr.to(device)
        train_atr_one = train_atr_one.to(device)

        f_code, f_label = make_label(10, args.batch)
        f_code, f_label = f_code.to(device), f_label.to(device)

        # print('train_atr',train_atr)
        # print('train_atr_one',train_atr_one)
        # print('f_code',f_code)
        # print('f_label',f_label)

##########################trainD##########################

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ ,out_feature= generator(train_img, f_code)
        rec_img, _,_ = generator(fake_img,train_atr_one)
        # flops = FlopCountAnalysis(g_ema, (fake_img,train_atr_one, ))
        # print("FLOPs: ", str(flops.total()/1000**3)+'G')
        # print(parameter_count_table(g_ema))

        fake_pred,_ = discriminator(fake_img, f_label)
        rec_pred,_ = discriminator(rec_img, train_atr)

        real_pred,real_cls = discriminator(train_img, train_atr)
        # print('real_cls',real_cls)
        # print('fake_pred',fake_pred) 
        c_loss = F.cross_entropy(real_cls,train_atr)
        # cr_loss = F.cross_entropy(real_cls,train_atr)
        # meanloss, varloss = criterion1(real_cls,train_atr)
        # c_loss = meanloss+varloss+cr_loss
        d_loss = d_logistic_loss(real_pred, fake_pred)+d_logistic_loss(real_pred, rec_pred)+c_loss
        # d_loss = d_logistic_loss(real_pred, fake_pred)+d_logistic_loss(real_pred, rec_pred) 
        loss_dict['d'] = d_loss
        loss_dict['c'] = c_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()
        # print(i,d_optim)
        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            train_img.requires_grad = True
            real_pred,_ = discriminator(train_img, train_atr)
            r1_loss = d_r1_loss(real_pred, train_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss
##########################trainG###########################
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _,_ = generator(train_img, f_code)
        rec_img, _,_ = generator(fake_img,train_atr_one)

        _, _ ,out_feature_train= generator(train_img, train_atr_one)
        _, _,out_feature_fake = generator(fake_img,f_code)
        _, _ ,out_feature_rec= generator(rec_img, train_atr_one)

        fake_pred,fake_cls = discriminator(fake_img, f_label)
        rec_pred,rec_cls = discriminator(rec_img,train_atr)
        fc_loss = F.cross_entropy(fake_cls,f_label,weight=ageclass_weights)
        rc_loss = F.cross_entropy(rec_cls,train_atr)
        # fcr_loss = F.cross_entropy(fake_cls,f_label)
        # fmeanloss, fvarloss = criterion1(fake_cls,f_label)
        # fc_loss = fmeanloss+fvarloss+fcr_loss
        # rcr_loss = F.cross_entropy(rec_cls,train_atr)
        # rmeanloss, rvarloss = criterion1(rec_cls,train_atr)
        # rc_loss = rmeanloss+rvarloss+rcr_loss
        # adaface = AdaFace(
        #     embedding_size=args.latent,
        #     classnum=10,
        #     m=0.4, h=0.333, s=64.0, t_alpha=0.01
        # ).to(device)
        # fake_resize = nn.functional.interpolate(fake_img, size=(112, 112))
        # real_resize = nn.functional.interpolate(train_img, size=(112, 112))
        # id_loss = mse_loss(identity(real_resize), identity(fake_resize))
        
        adaface = AdaFace(
            embedding_size=args.latent,  # 512，整数
            classnum=10,                 # 10 类，也是整数
            m=0.4, h=0.333, s=64.0, t_alpha=0.01
        ).to(device)
        l2_loss = mse_loss(train_img, fake_img)
        rec_loss = l1_loss(train_img,rec_img)
        norms_real = out_feature_train.norm(p=2, dim=1, keepdim=True)   # [B,1]
        norms_fake = out_feature_fake.norm(p=2, dim=1, keepdim=True)   # [B,1]
        emb_real = out_feature_train.mean(dim=[2,3], keepdim=False)  # [B,512]
        emb_fake = out_feature_fake .mean(dim=[2,3], keepdim=False)  # [B,512]
        embs = torch.cat([emb_real, emb_fake], dim=0)      # [2B,512]
        norms_real = emb_real.norm(p=2, dim=1, keepdim=True) 
        norms_fake = emb_fake.norm(p=2, dim=1, keepdim=True)
        train_label = train_atr.long()
        fake_label   = f_label.long()
        all_norms = torch.cat([norms_real, norms_fake], dim=0)  # [2B,1]
        all_labels = torch.cat([train_label, fake_label],    dim=0)      # [2B]
        # id_loss = l1_loss(out_feature_train,out_feature_fake)
        
        logits = adaface(embs, all_norms, all_labels)  # [2B, classnum]
        id_loss = F.cross_entropy(logits, all_labels)
        # fake_img, rec_img, feat_real, feat_fake, feat_rec, id_loss = compute_id_loss(
        #     generator,
        #     train_img,
        #     train_atr_one,
        #     f_code,
        #     train_atr,   # 真实年龄/身份标签 [B]
        #     adaface,
        #     device,
        # )
        pdist = PairwiseDistance(2)
        pos_dist = pdist.forward(out_feature_train, out_feature_rec)
        neg_dist = pdist.forward(out_feature_train, out_feature_fake)
        #pos_dist = pdist.forward(train_img, rec_img)
        ##neg_dist = pdist.forward(train_img, fake_img)
        hinge_dist = torch.clamp(0.5 +pos_dist - neg_dist, min=0.0)
        triplet_loss = torch.mean(hinge_dist)

        # loss_base_line = g_nonsaturating_loss(fake_pred)+ g_nonsaturating_loss(rec_pred) + fc_loss*1 + rc_loss*1 + l2_loss*10
        # g_loss =  loss_base_line + 0.1*triplet_loss + rec_loss*10 + id_loss
        perc_loss = perceptual_loss_fn(train_img, fake_img)
        #######################
        g_loss = g_nonsaturating_loss(fake_pred) + g_nonsaturating_loss(rec_pred) + fc_loss * 5 + id_loss * 5 + rec_loss * 10 + perc_loss
        #######################

        loss_dict['g'] = g_loss
        loss_dict['id'] = id_loss           #Lid
        loss_dict['l2'] = l2_loss           #Lpx 
        loss_dict['fc'] = fc_loss           #La
        loss_dict['rc'] = rc_loss
        loss_dict['rec'] = rec_loss         #Lcyc
        loss_dict['trip']=triplet_loss      #Lt
        loss_dict['perc'] = perc_loss     #Lperceptual 

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()
        # print(i,g_optim)
        # g_regularize = i % args.g_reg_every == 0

        # if g_regularize:
        #     # path_batch_size = max(1, args.batch // args.path_batch_shrink)
        #     # noise = mixing_noise(
        #         # path_batch_size, args.latent, args.mixing, device
        #     # )
        #     fake_img, latents, out_feature = generator(train_img, f_code, return_latents=True)

        #     path_loss, mean_path_length, path_lengths = g_path_regularize(
        #         fake_img, latents, mean_path_length
        #     )

        #     generator.zero_grad()
        #     weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

        #     if args.path_batch_shrink:
        #         weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        #     weighted_path_loss.backward()

        #     g_optim.step()

        #     mean_path_length_avg = (
        #         reduce_sum(mean_path_length).item() / get_world_size()
        #     )

        # loss_dict['path'] = path_loss
        # loss_dict['path_length'] = path_lengths.mean()      #Lpl

        #######################
        path_loss_val = 0
        path_length_val = 0
        #######################


        accumulate(g_ema, g_module, accum)
        # if get_rank() == 0 and (i % 5000 == 0):
        #     # 取一批验证图（已经用 sample_data 包装好无限循环）
        #     batch = next(val_loader)
        #     # 假设你的 Dataset 其实返回 (img, label) 两项，那么
        #     val_imgs   = batch[0]        # [B, C, H, W]
        #     val_labels = batch[1]        # [B]
        #     val_imgs = val_imgs.to(device)
        #     # 构造 one-hot code，假设 train_atr_one 逻辑相同
        #     val_code = torch.from_numpy(np.eye(10)[val_labels]).float().to(device)

        #     # 用 EMA generator 做推理
        #     with torch.no_grad():
        #         g_ema.eval()
        #         fake_val, _, _ = g_ema(val_imgs, val_code)

        #     # 拼接：真实 vs 生成
        #     # 把 real 和 fake 交叉成一张网格图
        #     comparison = torch.cat([val_imgs, fake_val], dim=0)  # [2B, C, H, W]
        #     save_path = f"sample/val_{str(i).zfill(6)}.png"
        #     utils.save_image(
        #         comparison,
        #         save_path,
        #        nrow=val_imgs.size(0),
        #        normalize=True,
        #        value_range=(-1, 1),
        #     )
        #     print(f"[VAL] step {i}: saved {save_path}")
        #     g_ema.train()
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        # path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        # path_length_val = loss_reduced['path_length'].mean().item()
        id_val = loss_reduced['id'].mean().item()
        l2_val = loss_reduced['l2'].mean().item()
        rec_val = loss_reduced['rec'].mean().item()
        trip_val = loss_reduced['trip'].mean().item()
        c_val = loss_reduced['c'].mean().item()
        fc_val = loss_reduced['fc'].mean().item()
        rc_val = loss_reduced['rc'].mean().item()
        perc_loss_val = loss_reduced['perc'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f};l2:{l2_val:.4f}; '
                    f' rec:{rec_val:.4f};c: {c_val:.4f};trip:{trip_val:.4f}; f_c: {fc_val:.4f}; r_c: {rc_val:.4f}; perc: {perc_loss_val:.4f}; '
                    f'path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}'
                )
            )


            if wandb and args.wandb:
                wandb.log(
                    {
                        'Generator': g_loss_val,
                        'Discriminator': d_loss_val,
                        'R1': r1_val,
                        'Path Length Regularization': path_loss_val,
                        'Mean Path Length': mean_path_length,
                        'Real Score': real_score_val,
                        'Fake Score': fake_score_val,
                        'Path Length': path_length_val,
						'ID': id_val,
						'L2': l2_val,
                        'rec': rec_val,
                        'trip':trip_val,
                        'Fake Class': fc_val,
 			            'Real Class': c_val,
                        'Rec Class': rc_val,
                        'Perceptual Loss': perc_loss_val,
                    }
                )

            if i % 1000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    old_code = make_specific_label(9, 10, sample_z.size(0)).to(device)
                    sample, _,_ = g_ema(sample_z, old_code)
                    utils.save_image(
                        sample,
                        f'sample/{str(i).zfill(6)}_old.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    yg_code = make_specific_label(0, 10, sample_z.size(0)).to(device)
                    sample, _,_ = g_ema(sample_z, yg_code)
                    utils.save_image(
                        sample,
                        f'sample/{str(i).zfill(6)}_young.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

                    sample,_,out_feature = g_ema(fake_img, train_atr_one)
                    utils.save_image(
                        sample,
                        f'sample/{str(i).zfill(6)}_rec.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    f'checkpoint/{str(i).zfill(6)}.pt',
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default="train_128_balance")
    parser.add_argument('--val_path', type=str, default="val_128_relabel_ori")
    parser.add_argument('--iter', type=int, default=750000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--n_sample', type=int, default=8)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    # parser.add_argument('--ckpt', type=str, default='checkpoint/600000.pt')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--wandb', default=True, action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--name', type=str, default='1048_ee303PC_wperc_0620')

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1
    perceptual_loss_fn = PerceptualLoss(layers=('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')).to(device)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)


    # ArcFace ID
    identity = IR_50([112, 112]).to(device)
    identity.load_state_dict(torch.load('./backbone_ir50_ms1m_epoch63.pth'))
    for param in identity.parameters():
       param.requires_grad = False

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr* g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        
        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            
        except ValueError:
            pass
            
        generator.load_state_dict(ckpt['g'],strict=False)
        discriminator.load_state_dict(ckpt['d'],strict=False)
        g_ema.load_state_dict(ckpt['g_ema'],strict=False)
        g_optim.load_state_dict(ckpt['g_optim'])
        # print(g_optim)
        d_optim.load_state_dict(ckpt['d_optim'])
        # print(d_optim)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    if not os.path.exists('sample'):
        os.makedirs('sample')
        os.makedirs('checkpoint')

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.train_path, transform, args.size)
    train_loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )


    dataset = MultiResolutionDataset(args.val_path, transform, args.size)
    val_loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.login(key="e8751c3686f935c9d94ddf6e9d509664ded38e70")
        wandb.init(project='AgetransGAN', name=args.name)

    print('-----------------------------\n', 'Parm: ', count_param(g_ema))
    train(args, train_loader, val_loader, generator, discriminator, g_optim, d_optim, g_ema, device)
