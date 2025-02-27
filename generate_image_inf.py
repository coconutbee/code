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
# 使用 img2img API 產圖 (生成單張)
def generate_image_from_prompt(
    image_prefix,             # 用於輸出檔名的前綴
    init_image_base64,        # 原圖 base64
    canny_edge_base64,        # Canny 邊緣 base64
    prompt_text,              # Prompt 文字
    input_file,               # e.g. "training"
    seed,                     # 隨機種子
    output_path               # 輸出路徑
):
    """
    調用 A1111 WebUI 的 /sdapi/v1/img2img，並將生成的圖片儲存到 output_path。
    """
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
        "seed": seed,
        "batch_size": 1,
        "batch_count": 1,
        "cfg_scale": 4.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "sgm_uniform",
        "denoising_strength": 0.7,
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
        # 回傳的 images[0] 是類似 "data:image/png;base64,xxx"
        result_str = r['images'][0]
        # 取出 base64 字串並轉為 PIL Image
        if "," in result_str:
            base64_part = result_str.split(",", 1)[1]
        else:
            base64_part = result_str

        generated_image = Image.open(io.BytesIO(base64.b64decode(base64_part)))

        # 確保輸出資料夾存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generated_image.save(output_path)
        print(f"圖像已保存：{output_path} (seed={seed})")
    else:
        print(f"生成圖像時發生錯誤：{response.status_code} - {response.text}")

#####################################
# 生成多張圖（對同一張原圖、多個 seed）
def generate_images_with_multiple_seeds(
    image_prefix,
    init_image_base64,
    canny_edge_base64,
    prompt_text,
    input_file,
    seeds
):
    """
    針對同一張原圖與同一個 prompt，多次呼叫 generate_image_from_prompt，
    每次使用不同的 seed 與輸出檔名。
    """
    for i, seed in enumerate(seeds, start=1):
        output_dir = f"Dataadd_0227/{input_file}"
        output_path = os.path.join(output_dir, f"{image_prefix}_seed{seed}_{i}.png")
        
        generate_image_from_prompt(
            image_prefix=image_prefix,
            init_image_base64=init_image_base64,
            canny_edge_base64=canny_edge_base64,
            prompt_text=prompt_text,
            input_file=input_file,
            seed=seed,
            output_path=output_path
        )

#####################################
if __name__ == '__main__':
    # 假設我們有一個資料夾存放對應的 prompt 檔(.txt)
    # 每個 txt 檔的檔名 prefix 與對應圖片前綴相同
    prompt_folder = "0227image_inf/prompt"
    prompts_dict = read_prompts_from_folder(prompt_folder)
    input_folder = "/media/avlab/reggie/Paul_sd35/stable-diffusion-webui-forge/0227"
    
    while True:
        seeds_to_use = [random.randint(0, 2**32-1) for _ in range(3)]  # 產生 3 個不同的 seed

        # 走訪要產圖的資料夾 (e.g. "cfp_fp/training")
        for file in os.listdir(prompt_folder):
            source_name = file.split("_")[0]
            print(f"source_name: {source_name}")
            process_folder = os.path.join(input_folder, source_name) 
            print(f"process_folder: {process_folder}")
            
            for file in os.listdir(process_folder):
                if file.lower().endswith(('.png', '.jpg')):  # 確保是圖片
                    file_prefix = os.path.splitext(file)[0]
                    file_prefix = os.path.join(f"{source_name}_{file_prefix}")
                    # print(f"file_prefix: {file_prefix}")
                    # print(f"file: {file}")
                if file_prefix not in prompts_dict:
                    print(f"找不到對應的 prompt，跳過。")
                    continue
                prompt_text = prompts_dict[file_prefix]
                print(f"prompt_text: {prompt_text}")
                image_path = os.path.join(process_folder, file)
                # print(f"image_path: {image_path}")

                # print(f"[處理中] 資料夾: {sub_folder_name}, 檔名: {file_prefix}")

                # 產生 Canny 邊緣
                # canny_edge_base64 = generate_canny_edge(image_path)
                canny_edge_base64 = os.path.join(input_folder, f"canny_{source_name}")
                if canny_edge_base64 is None:
                    print("Canny edge 產生失敗，跳過。")
                    continue

                # 讀取原圖為 base64
                with open(image_path, "rb") as imgf:
                    init_image_base64 = base64.b64encode(imgf.read()).decode("utf-8")

                # 使用多個 seed 產生多張圖
                generate_images_with_multiple_seeds(
                    image_prefix=file_prefix,
                    init_image_base64=init_image_base64,
                    canny_edge_base64=canny_edge_base64,
                    prompt_text=prompt_text,
                    input_file=source_name,
                    seeds=seeds_to_use
                )
