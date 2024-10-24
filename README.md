<p align="center">
  <h1 align="center"><img src="assets/logo.png" width="256"></h1>
  <h1 align="center">MIA-DPO: Multi-Image Augmented Direct Preference Optimization For Large Vision-Language Models</h1>
    <p align="center">
    <a href="https://github.com/Liuziyu77"><strong>Ziyu Liu</strong></a>
    ·
    <a href="https://yuhangzang.github.io/"><strong>Yuhang Zang</strong></a>
    ·
    <a href="https://lightdxy.github.io/"><strong>Xiaoyi Dong</strong></a>
    ·
    <a href="https://panzhang0212.github.io/"><strong>Pan Zhang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=sJkqsqkAAAAJ"><strong>Yuhang Cao</strong></a>
    ·
    <a href="https://kennymckormick.github.io/"><strong>Haodong Duan</strong></a>
    ·
    <a href="https://conghui.github.io/"><strong>Conghui He</strong></a>
    ·
     <a href="http://yjxiong.me/"><strong>Yuanjun Xiong</strong></a>
    ·
     <a href="http://dahua.site/"><strong>Dahua Lin</strong></a>
    ·
     <a href="https://myownskyw7.github.io/"><strong>Jiaqi Wang</strong></a>
  </p>
<!--   <h2 align="center">Accepted By NeurIPS2024 Dataset and Benchmark Track!</h2> -->
  📖<a href="https://arxiv.org/abs/2410.17637">Paper</a> |🏠<a href="https://liuziyu77.github.io/MIA-DPO/">Homepage</a></h3>
<!--   |🤗<a href="https://huggingface.co/datasets/laolao77/MMDU">Huggingface</a></h3> -->
<div align="center"></div>
<p align="center">
  <p>
Visual preference alignment involves training Large Vision-Language Models (LVLMs) to predict human preferences between visual inputs. This is typically achieved by using labeled datasets of chosen/rejected pairs and employing optimization algorithms like direct preference optimization (DPO). Existing visual alignment methods, primarily designed for single-image scenarios, struggle to effectively handle the complexity of multi-image tasks due to the scarcity of diverse training data and the high cost of annotating chosen/rejected pairs. 
    
🌈We present Multi-Image Augmented Direct Preference Optimization (MIA-DPO), a visual preference alignment approach that effectively handles multi-image inputs. MIA-DPO mitigates the scarcity of diverse multi-image training data by extending single-image data with unrelated images arranged in grid collages or pic-in-pic formats, significantly reducing the costs associated with multi-image data annotations. Our observation reveals that attention values of LVLMs vary considerably across different images. We use attention values to identify and filter out rejected responses the model may have mistakenly focused on. Our attention-aware selection for constructing the chosen/rejected pairs without relying on (i) human annotation, (ii) extra data, and (iii) external models or APIs. MIA-DPO is compatible with various architectures and outperforms existing methods on five multi-image benchmarks, achieving an average performance boost of 3.0% on LLaVA-v1.5 and 4.3% on the recent InternLM-XC2.5. Moreover, MIA-DPO has a minimal effect on the model's ability to understand single images.  
  </p>
    <a href="">
      <img src="assets/teaser.png" alt="Logo" width="100%"> 
    </a>
<br>

## 📢 News
- 🚀 [10/24/2024] We upload our paper to arxiv.
- 🚀 [10/24/2024] We release our DPO training code on github.

## 💡 Highlights
- 🔥 **Multi-Image Visual Alignment Pipeline**: We first design a multi-image visual alignment pipeline MIA-DPO. Our MIA-DPO requires no manual annotations and does not rely on APIs from larger models, offering a significant cost advantage compared to existing visual alignment approaches. 
- 🔥 **Observation on Multi-Image Hallucinations**: We contribute to the study of different types of multi-image hallucinations and propose to use attention values as an indicator for detecting multi-image hallucinations.
- 🔥 **Excellent Performance**:  MIA-DPO is agnostic to different LVLM architectures (LLaVA-v1.5 and InternLM-XC2.5, boosts the performance on multiple multi-image benchmarks while maintaining the original single-image understanding capabilities.

## 💩 Multi-Image Hallucinations
Some previous studies have explored different types of single-image hallucinations, such as object hallucination which means the model incorrectly describes objects that are not present in the image. Compared to single-image hallucinations, multi-image scenarios introduce more complex types of hallucinations. As shown in Fig. 2, we categorize multi-image hallucinations into two-types:

(1) ***Sequence Confusion.*** When presented with multiple images, the model may fail to identify which image the input prompt refers to. For instance, in the top case shown in Fig. 2, the question is directed at Image 3 (birds and sky), but the model responds based on Image 4 (a train on tracks).

(2) ***Element Interference.*** The presence of multiple images significantly increases the number of visual elements compared to a single image, leading to confusion between different elements by LVLMs. For example, in the bottom case of Fig. 2, the question “What color is the car in Image2?” should be answered with “white”. However, the LVLM incorrectly interpreted the color attribute of the motorcycle in Image 3 as the color of the car in Image 2, resulting in an incorrect response.

<a href="">
  <img src="assets/hallu.png" alt="Logo" >
</a>


## 💎 MIA-DPO Framework
**Attention as an Indicator for Detecting Hallucinations** The attention mechanism reveals wherethe model is “looking” when making a decision. We observe that the attention mechanism provides crucial clues for detecting multi-image hallucinations (Fig. 2). Ideally, attention values should focus on areas of the referred input image relevant to the question. If the attention values are scattered or not strongly focused on the correct visual element or region, it suggests the model is experiencing difficulty understanding multi-image sequences or distinguishing elements between different images. Based on our observation, we design an attention-aware selection that uses the attention values to select the rejected sample that contains the hallucinations in the DPO algorithm.

<a href="">
  <img src="assets/framework.png" alt="Logo">
</a>

**Post-Selection for Data Cleaning** Although our attention-aware selection is effective in constructing the DPO data, a small amount of noisy samples may be included and potentially causing detrimental effects. To filter out the noisy samples, we incorporate a post-selection step using the following three metrics: (1) ***Perplexity (PPL)*** (2) ***Length Ratio***  (3) ***Edit Distance***.

## 🎆 MIA-DPO Data Examples
Rather than expending effort on collecting and annotating new multi-image prompts, we efficiently convert existing single-image datasets, such as LLaVA-665k, by incorporating unrelated images. Our low-cost, scalable approach enriches data forms and allows us to comprehensively explore the various types of multi-image hallucinations that LVLMs might produce. As shown in Fig. 4, we construct multi-image prompts in three formats: (1) ***Sequence***: Multiple images are arranged sequentially, with questions targeting specific images. The number of images varies from 2 to 5. (2) ***Grid Collage***: Multiple images are merged into a single image, each labeled with a number description. Questions focus on specific images based on language descriptions. The number of images ranges from 2 to 9. (3) ***Pic-in-Pic***: One image is resized and overlaid onto another, and questions are asked about the combined image.

<a href="">
  <img src="assets/data_case.png" alt="Logo">
</a>


## 🛠️ Usage
### DPO Data Generation
We have provided users with the code to autonomously generate MIA-DPO data.

First, clone this project to your local machine and install the required dependencies. Then, enter the ``gen_data`` folder, where ``mix_data.ipynb``, ``make_data_pics_in_pics.ipynb``, and ``make_data_pingtu.ipynb`` are used to generate the three types of multi-image instruction data mentioned in the article: ***Sequence data, Grid Collage data, and Pic-in-Pic data.***
```
git clone https://github.com/Liuziyu77/MIA-DPO.git
cd ./gen_data
```

After that, you can run ``make_data_duotu.py``, ``make_data_pingtu.py``, and ``make_data_pip.py`` respectively to generate the corresponding Chosen and Rejected data for each type of instruction data.
```
python make_data_duotu.py
```
It should be noted that some model and file paths in the code need to be modified.

### Support Models
MIA-DPO currently supports models including the classic LLaVa-v1.5 and the recent InternLM-Xcomposer2.5. We have open-sourced the DPO code for LLaVa-v1.5, with the DPO code modified based on the LLaVa-Hound-DPO code.

### MIA-DPO for LLaVa1.5
We have modified the LLaVa-Hound-DPO code to support DPO training for the multi-image LLaVa-v1.5 model. When using it, please modify the parameters in ```MIA-DPO/LLaVA-Hound-DPO/llava_hound_dpo/dpo_scripts/train_dpo_multi.sh``` and run it. Note that the video-related parameters (e.g. video_dir) in the bash file can be ignored.
```
cd MIA-DPO/LLaVA-Hound-DPO/llava_hound_dpo/
bash ./dpo_scripts/train_dpo_multi.sh
```

### Evaluation
Evaluate the processed models using <a href="https://github.com/open-compass/VLMEvalKit"><strong>VLMEvalKit</strong></a>.


## ✒️Citation
```
TBD
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use
