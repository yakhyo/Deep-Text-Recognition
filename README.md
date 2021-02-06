# Deep Text Recognition

[Official](https://github.com/clovaai/deep-text-recognition-benchmark) PyTorch implementation of Deep text recognition | [Paper](https://arxiv.org/abs/1904.01906) |

## Run:
1. `git clone https://github.com/yakhyo/Deep-Text-Recognition`.
2. `mkdir weights`: make the `weights` folder inside the `Deep-Text-Recognition`.
3. Download pretrained weights from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)
4. Put some images into `images` folder.
5. Run the `image_folder_pred.py` file.
### Result:
Input image:<br> ![This is an input](./images/img_1.png)<br>
Prediction:<br>
```
--------------------------------------------------------------------------------
image_path               	predicted_labels         	confidence score
--------------------------------------------------------------------------------
./images/img_1.png       	namangan                 	0.9999
```
Input image:<br> ![This is an input](./images/img.png)<br>
Prediction:<br>
```
--------------------------------------------------------------------------------
image_path               	predicted_labels         	confidence score
--------------------------------------------------------------------------------
./images/img.png         	uzbekistan               	0.6281
```
## Fine-Tuning
1. Create your own lmdb dataset:
```python
pip install fire
python utils/create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```
The structure of data folder as below.
```
data
├── gt.txt
└── train
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
    └── ...
```
At this time, `gt.txt` should be `{imagepath}\t{label}\n`
For example
```
test/word_1.png Tiredness
test/word_2.png kills
test/word_3.png A
...
```
## Notice:
1. If you want to use pretrained weight for single image, use `single_image_pred.py` and specify `img_path` and run the file.
2. This repo is only for `TPS-ResNet-BiLSTM-Attn`.
- Thanks to the Clova AI Research team
## Reference

1. [Deep-Text-Recognition-Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

