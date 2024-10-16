import json
from tqdm import tqdm
import os
import random
import io
import os
import re
import seaborn as sns
import pdb
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
torch.manual_seed(1234)
import cv2
from PIL import Image

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

def plot_images(image_paths):
    num_images = len(image_paths)
    
    # 创建图形并显示图片
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from io import BytesIO


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image



def load_images(image_files):
    out = []
    for image_file in image_files:
        if isinstance(image_file, Image.Image):
            image = image_file.convert("RGB")
        else:
            image = load_image(image_file)
        out.append(image)
    return out


def eval_model(query, image_files, model, model_name, conv_mode=None, sep = ",", temperature=0, top_p=None, num_beams=1, max_new_tokens=1024):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # pdb.set_trace()
    conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    ### 四张图堆叠为 [4, 3, 336, 336]
    ### test = image_processor(images, return_tensors='pt')['pixel_values']
    ### print(test.shape)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # print(images_tensor.shape)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # pdb.set_trace()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    # pdb.set_trace()
    attention = output_ids.attentions
    output_scores = output_ids.scores
    # print("Attention Length: " + str(len(attention[0][0][0][0])))

    ### 返回attention, 所以用output_ids[0]而不是output_ids
    outputs = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True)[0].strip()
    return outputs, input_ids, attention, output_scores

disable_torch_init()

# model_path = "/mnt/petrelfs/liuziyu/V3Det/LLaVA/models/dpo_lora_llava_ori_dpo_28k/"
# model_base = "/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/llava-v1.5-7b/"
# model_name = get_model_name_from_path(model_path)
# print(model_name)

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path, model_base, model_name
# )


### 原模型代码     
model_path = "/mnt/hwfile/mllm/liuziyu/finetune_LLaVa/llava-v1.5-7b"
model_base = None
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)

random.seed(77)

with open("/mnt/petrelfs/liuziyu/RLHF/make_data/data_randomsample_randompic_45k/sft_data/mix_llava62k.json", 'r', encoding='utf-8') as file:
    llava62k = json.load(file)
random.shuffle(llava62k)
print(len(llava62k))

with open("/mnt/petrelfs/liuziyu/RLHF/make_data/data_randomsample_randompic_45k/sft_data/ping_llava187k_90k.json", 'r', encoding='utf-8') as file:
    ping_llava187k_90k = json.load(file)
random.shuffle(ping_llava187k_90k)
print(len(ping_llava187k_90k))

with open("/mnt/petrelfs/liuziyu/RLHF/make_data/data_randomsample_randompic_45k/sft_data/pics_in_pics_49k.json", 'r', encoding='utf-8') as file:
    pics_in_pics_49k = json.load(file)
random.shuffle(pics_in_pics_49k)
print(len(pics_in_pics_49k))

with open("/mnt/petrelfs/liuziyu/RLHF/make_data/data_randomsample_randompic_45k/sft_data/negative_text_llava62k_v4.json", 'r', encoding='utf-8') as file:
    negative_text_llava62k_v4 = json.load(file)
random.shuffle(negative_text_llava62k_v4)
print(len(negative_text_llava62k_v4))


def calculation_attention(linput_ids, attention, is_draw, layer_number=16):
    ### 设定输出的attention层数
    layer_number = layer_number
    ### 找到所有 <image> 的起始位置 
    indices = torch.nonzero(input_ids[0] == -200, as_tuple=False).squeeze().tolist()
    ### 输入单张图 indices 是一个 int
    if type(indices) == int:
        indices = [indices]
    ### 计算最后一张图像之后的 query 的 tokens 长度
    query_input_token_length = len(input_ids[0])-indices[-1]-1
    ### 换算插入image token之后每张图的起始位置 
    new_indices = []
    for i, item in enumerate(indices):
        if i == 0:
            new_indices.append(item)
        else:
            new_indices.append(item-i+576*i)
    indices = new_indices
    # print(indices)
    
    ### input_attention 输入的图文的 attention
    input_attention = torch.mean(attention[0][layer_number][0], dim=0)
    ### 输出的 tokens 的长度
    output_token_length = len(attention)-1
    ### 输出的 tokens 的 attention
    output_token_attention = []
    for i in range(output_token_length):
        output_token_attention.append(attention[i+1][layer_number][0])

    ### 计算 input tokens 和 outpu tokens 组成的总的 attention map 的大小 
    max_tesor_size = len(output_token_attention) + len(attention[0][layer_number][0][0])
    ### 把所有 attention merge 在一起
    merged_tensor = torch.zeros(len(attention[0][layer_number][0]), max_tesor_size, max_tesor_size)
    merged_tensor[:, :len(attention[0][layer_number][0][0]), :len(attention[0][layer_number][0][0])] = input_attention
    # 依次拼接大小为torch.Size([32, 1, N])的tensor
    for i in range(1, len(output_token_attention)+1):
        merged_tensor[:, len(attention[0][layer_number][0][0]) + i - 1, :len(attention[0][layer_number][0][0]) + i] = output_token_attention[i-1].squeeze(1)
    merged_tensor = torch.mean(merged_tensor, dim=0)
    # print(merged_tensor.shape)

    ### 获取图像的 image_attention
    image_tensors = []
    for i in range(len([images])):
        image_tensors.append(merged_tensor[-(len(output_token_attention)):, indices[i]:indices[i]+576])
    mean_image_tensors = [torch.mean(image_tensor, dim=0) for image_tensor in image_tensors]
    reshape_image_tensors = [mean_image_tensor.reshape(24, 24) for mean_image_tensor in mean_image_tensors]
    tensors = [reshape_image_tensor.cpu() for reshape_image_tensor in reshape_image_tensors]
    

    final_tensor = torch.cat(tensors, dim = 1)
    final_tensor_np = final_tensor.numpy()
    # print(final_tensor_np.shape)

    if is_draw==True:
        plt.figure(figsize=(10, 10))
        cax = plt.imshow(final_tensor_np, cmap='viridis', interpolation='nearest', vmin=0, vmax=0.0012)
        cbar = plt.colorbar(cax, fraction=0.036, pad=0.04, aspect=10)
        cbar.set_label('Intensity')
        plt.title('Heatmap of Merged Tensor Matrix')
        plt.show()

    return final_tensor_np


def split_and_calculate_ratio(array, split_dims, block_index):
    rows, cols = split_dims
    
    # 计算每个块的大小
    block_height = array.shape[0] // rows
    block_width = array.shape[1] // cols
    
    # 处理不能整除的情况
    block_heights = [block_height] * rows
    block_widths = [block_width] * cols
    
    for i in range(array.shape[0] % rows):
        block_heights[i] += 1
        
    for j in range(array.shape[1] % cols):
        block_widths[j] += 1
    
    # 累计高度和宽度，确定块的边界
    height_cumsum = np.cumsum([0] + block_heights)
    width_cumsum = np.cumsum([0] + block_widths)
    
    # 确定指定块的行列位置
    row_idx = (block_index - 1) // cols
    col_idx = (block_index - 1) % cols
    
    # 切割指定的块
    block = array[height_cumsum[row_idx]:height_cumsum[row_idx + 1], width_cumsum[col_idx]:width_cumsum[col_idx + 1]]
    
    # 计算指定块的和和整个数组的和
    block_sum = np.sum(block)
    total_sum = np.sum(array)
    
    # 计算比例
    ratio = block_sum / total_sum
    
    return block_sum, ratio


def add_gaussian_noise_to_region(image_path, grid_size, region_index, noise_std=64):
    
    image = cv2.imread(image_path)
    
    rows, cols = grid_size
    height, width = image.shape[:2]
    
    # 计算每个区域的高度和宽度
    region_height = height // rows
    region_width = width // cols
    
    # 计算区域的行列索引
    region_row = (region_index - 1) // cols
    region_col = (region_index - 1) % cols
    
    # 计算区域的起始点和结束点
    start_y = region_row * region_height
    start_x = region_col * region_width
    end_y = start_y + region_height
    end_x = start_x + region_width
    
    # 提取指定区域
    region = image[start_y:end_y, start_x:end_x]
    
    # 生成与区域大小相同的高斯噪声
    noise = np.random.normal(0, noise_std, region.shape).astype(np.float32)
    
    # 将噪声添加到该区域
    noisy_region = cv2.add(region.astype(np.float32), noise)
    
    # 将加噪后的区域放回图像
    noisy_image = image.copy()
    noisy_image[start_y:end_y, start_x:end_x] = np.clip(noisy_region, 0, 255).astype(np.uint8)

    noisy_image_pil = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    
    return noisy_image_pil

def compute_softmax_max_and_ppl(output_scores):
    max_values = []

    for tensor in output_scores:
        # 计算softmax值
        softmax_vals = torch.softmax(tensor[0], dim=0)
        # 获取softmax结果的最大值
        max_val = torch.max(softmax_vals)
        max_values.append(max_val.item())  # 将Tensor转换为Python数值类型

    # 使用最大softmax值计算PPL
    log_probs = [torch.log(torch.tensor(max_val)) for max_val in max_values]  # 计算对数概率
    avg_log_prob = sum(log_probs) / len(log_probs)  # 计算平均对数概率
    ppl = torch.exp(-avg_log_prob).item()  # 计算PPL

    return max_values, ppl

### 没有sft步骤直接使用全部数据
selected_pingtu_data = ping_llava187k_90k
local_index = 7
selected_pingtu_data = selected_pingtu_data[78750:]
print(len(selected_pingtu_data))

dpo_data = []
count = 0
for item in tqdm(selected_pingtu_data):
    try:
        ### 预处理数据
        images = item['image']
        question = item['conversations'][0]['value'].replace("<image>", "<image-placeholder>")
        gt = item['conversations'][1]['value']
        if len(gt)<512 :
            ### 首先计算拼图数据中有几张图
            w, l = Image.open(images).size
            l_num = l//448
            w_num = w//448
            # print(l_num, w_num)
            all_pic_number = l_num*w_num
    
            ### 提取图片的index
            match = re.search(r'In Image(\d+)', question)
            if match:
                extracted_str = match.group(0)
                extracted_number = match.group(1)
            # print("Image index: "+str(extracted_number))
    
            ### 模糊图片
            blur_images = add_gaussian_noise_to_region(images, (l_num, w_num), int(extracted_number), noise_std=225)
        
            ### 推理
            with torch.cuda.amp.autocast():
                response, input_ids, attention, output_scores = eval_model(question, [blur_images], model, model_name, max_new_tokens= 512)
            # print(GREEN+response+RESET)
    
            ### 计算attention
            final_tensor_np = calculation_attention(input_ids,attention,False)
    
            # 计算最大softmax值,计算ppl
            max_softmax_values, ppl = compute_softmax_max_and_ppl(output_scores)
            # print(max_softmax_values, ppl)
            
            ### attention 根据图像的数量进行切块计算
            _, ratio = split_and_calculate_ratio(final_tensor_np, [l_num, w_num], int(extracted_number))
            # print(ratio)
        
            ### 显示低于阈值的数据
            if gt!=response:
                if all_pic_number == 2 and ratio<0.7 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = [item['image']]
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
                elif all_pic_number == 3 and ratio<0.6 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = [item['image']]
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
                elif all_pic_number == 4 and ratio<0.5 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = [item['image']]
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
                elif all_pic_number == 6 and ratio<0.4 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = [item['image']]
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
                elif all_pic_number == 9 and ratio<0.3 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = [item['image']]
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
                
    except Exception as e:
        print({e})

file_path = f"./pingtu_{local_index}.json"
print(len(dpo_data))
with open(file_path, 'w') as file:
    json.dump(dpo_data, file, indent=4)
print(f'数据已保存到 {file_path}')