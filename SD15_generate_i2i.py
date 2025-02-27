import io
import cv2
import base64
import requests
import os
from PIL import Image
import random

#####################################
# A1111 WebUI 位置
url = "http://127.0.0.1:7860"

#####################################
# 讀取指定資料夾所有的文本檔(.txt)，
# 以字典形式回傳 { prefix: prompt_text }
def read_prompts_from_folder(folder_path):
    prompts_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            prefix = os.path.splitext(file_name)[0]  # 前綴(不含副檔名)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            prompts_dict[prefix] = prompt
    return prompts_dict

#####################################
# 產生 Canny 邊緣圖 (回傳 base64 編碼後的字串)
def generate_canny_edge(input_image_path, low_threshold=100, high_threshold=200):
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"無法讀取圖像：{input_image_path}")
        return None

    edges = cv2.Canny(img, low_threshold, high_threshold)
    # 將 edges 圖轉成 base64
    _, buffer = cv2.imencode('.png', edges)
    encoded_edges = base64.b64encode(buffer).decode('utf-8')
    return encoded_edges

#####################################
# 使用 img2img API 產圖
def generate_image_from_prompt(
    image_prefix,             # 用於檔名、識別的 prefix
    sub_folder_name,          # 子資料夾
    init_image_base64,        # 原圖 base64
    canny_edge_base64,        # Canny 邊緣 base64
    prompt_text,              # 對應文字 prompt
    input_file                # e.g. "training"
):
    # 隨機種子
    # seed = random.randint(0, 5000)
    # seed = 65,

    # 可以自行調整你的 negative_prompt 或 sampler 等參數
    payload = {
        "init_images": [init_image_base64],  # 原圖
        "prompt": prompt_text,               # 文字 prompt
        "negative_prompt": (
            "cropped, easynegative, censored, furry, 3d, photo, monochrome, elven ears, anime, "
            "extra legs, extra hands, mutated legs, mutated hands, extra fingers"
        ),
        "width": 512,
        "height": 512,
        "steps": 30,
        "seed": 65,
        "batch_size": 1,
        "batch_count": 1,
        "cfg_scale": 4.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "sgm_uniform",
        "denoising_strength": 0.9,
        "override_settings": {
            "sd_model_checkpoint": "v1-5-pruned-emaonly.safetensors"
        },
        # ControlNet 設定
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny [d14c016b]",
                        "weight": 1,  # 可自行調整
                        # 有些版本參數名稱可能不同，需依 WebUI 版本修改
                        "input_image": f"data:image/png;base64,{canny_edge_base64}",
                        "control_mode": 1
                    }
                ]
            }
        }
    }

    # 呼叫 A1111 WebUI API
    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

    if response.status_code == 200:
        r = response.json()
        # 回傳的 images[0] 是 "data:image/png;base64,xxx"
        result_str = r['images'][0]
        # 取出 base64 並轉為 PIL Image
        if "," in result_str:
            base64_part = result_str.split(",", 1)[1]
        else:
            base64_part = result_str

        generated_image = Image.open(io.BytesIO(base64.b64decode(base64_part)))

        # 輸出資料夾
        output_dir = f"Dataadd/{input_file}/{sub_folder_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 輸出檔名
        output_path = os.path.join(output_dir, f"{image_prefix}.png")
        generated_image.save(output_path)
        # print(f"生成的圖像已保存：{output_path}，隨機種子：{seed}")
    else:
        print(f"生成圖像時發生錯誤：{response.status_code} - {response.text}")


#####################################
if __name__ == '__main__':
    # 假設我們有一個資料夾存放對應的 prompt 檔(.txt)
    # 每個 txt 檔的檔名 prefix 與對應圖片前綴相同
    prompt_folder = "/media/avlab/micky/Paul/stable-diffusion-webui-forge/ICIP2025/cplfw_prompt_withdraw"   # 請自行調整

    # 讀取 prompts，回傳字典 { prefix: prompt_text }
    prompts_dict = read_prompts_from_folder(prompt_folder)

    # 走訪要產圖的資料夾 (e.g. "cfp_fp/training")
    input_folder = "/media/avlab/micky/Paul/stable-diffusion-webui-forge/ICIP2025/cplfw"
    input_file = os.path.basename(input_folder)  # e.g. "training"

    for parent_folder, sub_folders, files in os.walk(input_folder):
        sub_folder_name = os.path.basename(parent_folder)
        for file in files:
            if file.lower().endswith(('.png', '.jpg')):
                # 取前綴 (不含副檔名)
                file_prefix = os.path.splitext(file)[0]

                # 檢查有沒有對應的 prompt
                if file_prefix in prompts_dict:
                    prompt_text = prompts_dict[file_prefix]
                    
                    # 建立完整路徑
                    image_path = os.path.join(parent_folder, file)
                    print(f"[處理中] 資料夾: {sub_folder_name}, 檔名: {file_prefix}")

                    # 先產生 Canny 邊緣
                    canny_edge_base64 = generate_canny_edge(image_path)
                    if canny_edge_base64 is None:
                        print("Canny edge 產生失敗，跳過。")
                        continue

                    # 讀取原圖為 base64
                    with open(image_path, "rb") as imgf:
                        init_image_base64 = base64.b64encode(imgf.read()).decode("utf-8")

                    # 只產一張
                    generate_image_from_prompt(
                        image_prefix=file_prefix,
                        sub_folder_name=sub_folder_name,
                        init_image_base64=init_image_base64,
                        canny_edge_base64=canny_edge_base64,
                        prompt_text=prompt_text,
                        input_file=input_file
                    )
                else:
                    # 若沒有找到對應的 prompt，就略過
                    pass
