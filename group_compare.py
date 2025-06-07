import net
import torch
import os
import torch.nn.functional as F
from face_alignment import align
import numpy as np
import csv
from tqdm import tqdm

adaface_models = {
    'ir_50': "pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    assert architecture in adaface_models
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {k[6:]: v for k, v in statedict.items() if k.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    if pil_rgb_image is None:
        return None
    np_img = np.array(pil_rgb_image)
    bgr = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    arr = bgr.transpose(2, 0, 1).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def build_key_map(folder):
    m = {}
    for fname in os.listdir(folder):
        key = fname.split('_')[0]
        m[key] = fname
    return m


if __name__ == '__main__':
    model = load_pretrained_model('ir_50')
    root = '/media/avlab/disk_A/CAF/colored'

    # 迴圈 i 進度條
    for i in tqdm(range(17), desc="Processing age groups"):
        #05,15,25,35, 75,85,95 = 7 0123789
        # 設定資料夾與 CSV 路徑
        if i == 9:
            folderA = os.path.join(root, '0')
            folderB = os.path.join(root, '9')
            csv_path = f'{root}/csim_result/CSIM_09.csv'
        elif i < 9:
            folderA = os.path.join(root, str(i))
            b = i + 1
            folderB = os.path.join(root, str(b))
            csv_path = f'{root}/csim_result/CSIM_{i}{b}.csv'
        else: #10,11,12,13, 14,15,16
            b = i - 10 # 0,1,2,3,4,5,6
            folderA = os.path.join(root, '5')
            if b < 4: #0,1,2,3
                folderB = os.path.join(root, str(b))
                csv_path = f'{root}/csim_result/CSIM_{b}{5}.csv'
            else: # 4,5,6
                c = b + 3
                folderB = os.path.join(root, str(c))
                csv_path = f'{root}/csim_result/CSIM_{5}{c}.csv'
        mapA = build_key_map(folderA)
        mapB = build_key_map(folderB)

        common_keys = sorted(set(mapA.keys()) & set(mapB.keys()))
        if not common_keys:
            print(f"[Warn] i={i}: 沒有相同的 key 可比較！")
            continue

        results = []
        # 比較 common_keys 進度條
        for key in tqdm(common_keys, desc=f"Comparing {i}->{(i+1)%10}", leave=False):
            fA = mapA[key]
            fB = mapB[key]
            pathA = os.path.join(folderA, fA)
            pathB = os.path.join(folderB, fB)

            # 對齊 & 轉 tensor
            alignedA = align.get_aligned_face(pathA)
            alignedB = align.get_aligned_face(pathB)
            if alignedA is None or alignedB is None:
                continue

            inpA = to_input(alignedA)
            inpB = to_input(alignedB)
            if inpA is None or inpB is None:
                continue

            # 前向抽特徵
            featA, _ = model(inpA)
            featB, _ = model(inpB)
            vA = featA.squeeze(0)
            vB = featB.squeeze(0)

            # 計算 cosine similarity
            sim = F.cosine_similarity(vA, vB, dim=0).item()

            # 儲存結果
            results.append({
                'key': key,
                'fileA': fA,
                'fileB': fB,
                'similarity': sim
            })

        # 寫入 CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['key','fileA','fileB','similarity'])
            writer.writeheader()
            for row in results:
                writer.writerow(row)
            if results:
                avg_sim = sum(r['similarity'] for r in results) / len(results)
                f.write(f"\n成功比較 {len(results)} 對圖片，平均相似度：{avg_sim:.4f}\n")
            else:
                print(f"[Warn] i={i}: 沒有成功比較任何圖片。")

        print(f"結果已寫入 {csv_path}")
