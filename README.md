# T23DAQA
Official repo for "Multi-Dimensional Quality Assessment for Text-to-3D Assets: Dataset and Model" [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2502.16915).

## Introduction
Recent advancements in text-to-image (T2I) generation have spurred the development of text-to-3D asset (T23DA) generation, leveraging pretrained 2D text-to-image diffusion models for text-to-3D asset synthesis. Despite the growing popularity of text-to-3D asset generation, its evaluation has not been well considered and studied. However, given the significant quality discrepancies among various text-to-3D assets, there is a pressing need for quality assessment models aligned with human subjective judgments. 

To tackle this challenge, we conduct a comprehensive study to explore the T23DA quality assessment (T23DAQA) problem in this work from both subjective and objec- tive perspectives. Given the absence of corresponding databases, we first establish the largest text-to-3D asset quality assessment database to date, termed the AIGC-T23DAQA database. 

This database encompasses 969 validated 3D assets generated from 170 prompts via 6 popular text-to-3D asset generation models, and corresponding subjective quality ratings for these assets from the perspectives of quality, authenticity, and text-asset correspon- dence, respectively. 

## Database Description
The googledrive downloadlink is [here](https://drive.google.com/file/d/1UMAnmxMp9I4khzUGvPeo01Mq4IBJI6j7/view?usp=drive_link).

The **T23DAQA** database contains 2 types of files:

A. videos.zip 

The projection videos of AI-Generated 3D assests

B. anno.txt 

Path|Prompt|Quality|Authenticity|Correspondence

## License

The database is distributed under the MIT license.

## 3. Citation

If you find our work useful, please cite our paper as:
```
@misc{fu2025t23daqa,
      title={Multi-Dimensional Quality Assessment for Text-to-3D Assets: Dataset and Model}, 
      author={Kang Fu, Huiyu Duan, Zicheng Zhang, Xiaohong Liu, Xiongkuo Min, Jia Wang and Guangtao Zhai},
      year={2025},
      eprint={2502.16915},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Contact

For any queries or feedback, please reach out to [fuk20-20@sjtu.edu.cn].
