import torch
import torch.nn.functional as F
from AdaFace.head import AdaFace

def compute_id_loss(
    generator: torch.nn.Module,
    real_img: torch.Tensor,
    real_code: torch.Tensor,
    fake_code: torch.Tensor,
    labels: torch.LongTensor,
    adaface: AdaFace,
    device: torch.device,
) -> torch.Tensor:
    """
    1) 用 generator 生成 fake_img（fake_code）和 rec_img（real_code）；
    2) 提取 3 条特征：feat_real, feat_fake, feat_rec；
    3) 基于 feat_real vs feat_fake 用 AdaFace 计算 ID Loss。

    参数:
      generator   – 你的 Generator 实例
      real_img    – 真实图 [B,C,H,W]
      real_code   – 真实条件 one-hot [B,classnum]
      fake_code   – 生成条件 one-hot [B,classnum]
      labels      – 标签 [B]，0…classnum-1
      adaface     – 已实例化的 AdaFace
      device      – "cuda" 或 "cpu"
      loss_weight – 最后返回的 id_loss * loss_weight

    返回:
      feat_real, feat_fake, feat_rec, id_loss
    """

    # 1. 先生成 fake_img 和 rec_img
    fake_img, _, feat_fake = generator(real_img, fake_code)
    rec_img,  _, feat_rec  = generator(fake_img, real_code)

    # 2. 再用 real_code 提取真实图的特征
    _, _, feat_real = generator(real_img, real_code)

    # 3. AdaFace 只用 feat_real vs feat_fake
    #    L2 归一化
    norms_r = feat_real.norm(p=2, dim=1, keepdim=True)
    norms_f = feat_fake.norm(p=2, dim=1, keepdim=True)
    emb_r   = feat_real / norms_r
    emb_f   = feat_fake / norms_f

    #    拼 batch
    embs   = torch.cat([emb_r, emb_f], dim=0)     # [2B, D]
    norms  = torch.cat([norms_r, norms_f], dim=0) # [2B, 1]
    labs   = torch.cat([labels, labels], dim=0)   # [2B]

    #    计算 margin logits + CE
    logits = adaface(embs, norms, labs)
    id_loss = F.cross_entropy(logits, labs) 

    return fake_img, rec_img, feat_real, feat_fake, feat_rec, id_loss
