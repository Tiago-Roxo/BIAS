# Textual Interpretability

1. Download the ASD-related captions dataset [here](https://drive.google.com/file/d/1quJtGrFX4mS2DngIQt5dR14uDUZ0gG7M/view?usp=share_link) and extract the .zip to the `COCO` folder;
2. To finetune the ViT-GPT2 model, execute the command:
```bash
python3 train.py 
```
which will download the [pretrained model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) from HuggingFace and finetune him to the ASD-related captions dataset, in the `COCO/ASD_captions` folder;

3. To obtain ViT-GPT2 model predictions, execute the command:
```bash
python3 inference.py 
```
which will output the captions given by the previously finetuned model (`models` folder) for the images in the `imgs` folder. If you do not have any finetuned model, you can use the ASD-captions finetuned ViT-GPT2 in [here](https://drive.google.com/file/d/1Et86ERovrodPXY7U-zpwT491uQcslJiD/view?usp=share_link) and replace the name of the folder extracted from .zip (`pretrained_GPT_ASD`) to `models`.

For any additional issues/doubts regarding ViT-GPT2 use, go to the [official](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) HuggingFace page, which contains a detailed [tutorial](https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/).



## Cite

```bibtex
@misc{roxo2024bias,
      title={BIAS: A Body-based Interpretable Active Speaker Approach}, 
      author={Tiago Roxo and Joana C. Costa and Pedro R. M. Inácio and Hugo Proença},
      year={2024},
      eprint={2412.05150},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05150}, 
}
```
