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
import random

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


def calculation_attention(input_ids, attention, images, is_draw):
    ### 设定输出的attention层数
    layer_number = 16
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
    for i in range(len(images)):
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

    return final_tensor_np, tensors

def add_gaussian_noise(image, mean=0, std=25):
    np_image = np.array(image)
    gauss = np.random.normal(mean, std, np_image.shape).astype('uint8')
    noisy_image = np_image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = Image.fromarray(noisy_image.astype('uint8'))
    
    return noisy_image

def process_images_with_noise(image_paths, noise_std=25):
    temp_images = []
    for path in image_paths:
        image = Image.open(path)
        noisy_image = add_gaussian_noise(image, std=noise_std)
        temp_images.append(noisy_image)
    return temp_images

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


### 直接DPO所有数据都可以用, 62k/8 = 8k
selected_duotu_data = llava62k
local_index = 74
selected_duotu_data = selected_duotu_data[62000:]
print(len(selected_duotu_data))

dpo_data = []
count = 0
for item in tqdm(selected_duotu_data):
    try:
        images = item['image']
        images = ['/mnt/petrelfs/liuziyu/V3Det/LLaVA/scripts/v1_5/data/cl_data/' + path for path in images]
        # if len(images) >=5:
        question = item['conversations'][0]['value'].replace("<image>", "<image-placeholder>")
        gt = item['conversations'][1]['value']
        # plot_images(images)
        # print(RED+question+RESET)
        # print(GREEN+gt+RESET)

        if len(gt)<512:
            ### 提取图片的index
            match = re.search(r'In Image(\d+)', question)
            if match:
                extracted_str = match.group(0)
                extracted_number = match.group(1)
            # print("Image index: "+str(extracted_number))
        
            # blur_images = process_images_with_noise(images, noise_std=64)
            # fig, axes = plt.subplots(1, len(blur_images), figsize=(15, 5))
            # for ax, img in zip(axes, blur_images):
            #     ax.imshow(img)
            #     ax.axis('off')  # 不显示坐标轴
            # plt.show()
        
            with torch.cuda.amp.autocast():
                response, input_ids, attention, output_scores = eval_model(question, images, model, model_name, max_new_tokens= 512)
            # print(GREEN+response+RESET)
        
            final_tensor_np, tensors = calculation_attention(input_ids, attention, images, False)
            means = [torch.mean(tensor).item() for tensor in tensors]
            # print("Mean: ", means)
            ratio = means[int(extracted_number)-1]/sum(means)
            # print("ratio: ", ratio)

            # 计算最大softmax值,计算ppl
            max_softmax_values, ppl = compute_softmax_max_and_ppl(output_scores)
    
            ### 筛选显示数据
            if gt!=response:
                if len(images) ==2 and ratio<=0.8 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = item['image']
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
                elif len(images) ==3 and ratio<=0.7 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = item['image']
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
                elif len(images) ==4 and ratio<=0.6 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = item['image']
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
                elif len(images) ==5 and ratio<=0.6 and ppl>1.1:
                    count+=1
                    print(count)
                    print(ratio)
                    print(RED+question+RESET)
                    print(GREEN+gt+RESET)
                    print(GREEN+response+RESET)
                    new_dict = {}
                    new_dict['id'] = item['id']
                    new_dict['image'] = item['image']
                    new_dict["prompt"] = question
                    new_dict["chosen"] = gt
                    new_dict["rejected"] = response
                    new_dict["gt"] = gt
                    if new_dict["chosen"]!='' and new_dict["rejected"]!='' and new_dict["chosen"]!=new_dict["rejected"]:
                        dpo_data.append(new_dict)
        
    except Exception as e:
        print({e})

file_path = f"./duotu_{local_index}.json"
print(len(dpo_data))
with open(file_path, 'w') as file:
    json.dump(dpo_data, file, indent=4)
print(f'数据已保存到 {file_path}')