# FairMOT_distillation_knowledge
Use python 3.10.12
```
conda create -n MOT
conda activate MOT
conda install pytorch
cd ${FAIRMOT_ROOT}
pip install cython
pip install -r requirements.txt
```
## Data preparation

* **MOT17, MOT20 & Dancetrack** 
[MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/), [DanceTrack](https://github.com/DanceTrack/DanceTrack) can be downloaded from the official webpage of MOT challenge. After downloading, you should prepare the data in the following structure:

```
MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
MOT20
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)

Dancetrack
   |——————images
   |        └——————train
   |        └——————val
   └——————labels_with_ids
            └——————train(empty)
            └——————val(empty)

 Fish
   |——————images
   |        └——————train
   |        └——————val
   └——————labels_with_ids
            └——————train(empty)
            └——————val(empty)
```

Then, you can change the seq_root and label_root in src/gen_labels_17.py and src/gen_labels_20.py and run:

```
cd src
python gen_labels_17.py
python gen_labels_20.py
python gen_labels_dt.py
python gen_labels_fish.py
```

to generate the labels of 2DMOT15 and MOT20. The seqinfo.ini files of 2DMOT15 can be downloaded here [[Google]](https://drive.google.com/open?id=1kJYySZy7wyETH4fKMzgJrYUrTfxKlN1w), [[Baidu],code:8o0w](https://pan.baidu.com/s/1zb5tBW7-YTzWOXpd9IzS0g).

## Pretrained models and baseline model
* **Pretrained models**

HRNetV2 ImageNet pretrained model: [HRNetV2-W18 official](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw), [HRNetV2-W32 official](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w).
After downloading, you should put the pretrained models in the following structure:
```
${FAIRMOT_ROOT}
   └——————models
           └——————hrnetv2_w32_imagenet_pretrained.pth
           └——————hrnetv2_w18_imagenet_pretrained.pth
```

* **Baseline model**

Our baseline FairMOT model (DLA-34 backbone) is pretrained on the CrowdHuman for 60 epochs with the self-supervised learning approach and then trained on the MIX dataset for 30 epochs. The models can be downloaded here: 
crowdhuman_dla34.pth [[Google]](https://drive.google.com/file/d/1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG/view?usp=sharing) [[Baidu, code:ggzx ]](https://pan.baidu.com/s/1JZMCVDyQnQCa5veO73YaMw) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/EUsj0hkTNuhKkj9bo9kE7ZsBpmHvqDz6DylPQPhm94Y08w?e=3OF4XN).
fairmot_dla34.pth [[Google]](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view?usp=sharing) [[Baidu, code:uouv]](https://pan.baidu.com/s/1H1Zp8wrTKDk20_DSPAeEkg) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/EWHN_RQA08BDoEce_qFW-ogBNUsb0jnxG3pNS3DJ7I8NmQ?e=p0Pul1). (This is the model we get 73.7 MOTA on the MOT17 test set. )
After downloading, you should put the baseline model in the following structure:
```

## Training
* Download the training data
* Change the dataset root directory 'root' in src/lib/cfg/data.json and 'data_dir' in src/lib/opts.py
* Pretrain on CrowdHuman and train on MIX:
```
See snellius script folder
```
* Train on MOT20:
The data annotation of MOT20 is a little different from MOT17, the coordinates of the bounding boxes are all inside the image, so we need to uncomment line 313 to 316 in the dataset file src/lib/datasets/dataset/jde.py:
```
#np.clip(xy[:, 0], 0, width, out=xy[:, 0])
#np.clip(xy[:, 2], 0, width, out=xy[:, 2])
#np.clip(xy[:, 1], 0, height, out=xy[:, 1])
#np.clip(xy[:, 3], 0, height, out=xy[:, 3])
```

The ablation study model 'mix_mot17_half_dla34.pth' can be downloaded here: [[Google]](https://drive.google.com/file/d/1dJDGSa6-FMq33XY-cOd_nYxuilv30YDM/view?usp=sharing) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/ESh1SlUvZudKgUX4A8E3yksBhfRHIf2AsKaaPJ-v_5lVAw?e=NB6UHR) [[Baidu, code:iifa]](https://pan.baidu.com/s/1RQD8ik1labWuwd8jJ-0ukQ).
* Performance on the test set of MOT17 when using different training data:

| Training Data    |  MOTA | IDF1 | IDS     |
|--------------|-----------|--------|-------|
|MOT17  | 69.8 | 69.9 | 3996                |
|MIX       | 72.9 | 73.2 | 3345             |
|CrowdHuman + MIX     | 73.7 | 72.3 | 3303  |
* We use CrowdHuman, MIX and MOT17 to train the light version of FairMOT using yolov5s as backbone:
```
sh experiments/all_yolov5s.sh
```
The pretrained model of yolov5s on the COCO dataset can be downloaded here:  [[Google]](https://drive.google.com/file/d/1Ur3_pa9r3KRY-5qM2cdFhFJ5exghRJvh/view?usp=sharing) [[Baidu, code:wh9h]](https://pan.baidu.com/s/1JHjN_l1nkMnRHRF5TcHYXg).

The model of the light version 'fairmot_yolov5s' can be downloaded here:  [[Google]](https://drive.google.com/file/d/1MEvsRPyoAqYSCdKaS5Ofrl7ZfKbBZ1Jb/view?usp=sharing) [[Baidu, code:2y3a]](https://pan.baidu.com/s/1dyBEeiGpRfZhqae0c264rg).

## Tracking
* The default settings run tracking on the validation dataset from 2DMOT15. Using the baseline model, you can run:
```
See scripts

```

Results of the test set all need to be evaluated on the MOT challenge server. You can see the tracking results on the training set by setting --val_motxx True and run the tracking code. We set 'conf_thres' 0.4 for MOT16 and MOT17. We set 'conf_thres' 0.3 for 2DMOT15 and MOT20. 

## Demo
You can input a raw video and get the demo video by running src/demo.py and get the mp4 format of the demo video:
```
cd src
python demo.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.4
```
You can change --input-video and --output-root to get the demos of your own videos.
--conf_thres can be set from 0.3 to 0.7 depending on your own videos.

## Train on custom dataset
You can train FairMOT on custom dataset by following several steps bellow:
1. Generate one txt label file for one image. Each line of the txt label file represents one object. The format of the line is: "class id x_center/img_width y_center/img_height w/img_width h/img_height". You can modify src/gen_labels_16.py to generate label files for your custom dataset.
2. Generate files containing image paths. The example files are in src/data/. Some similar code can be found in src/gen_labels_crowd.py
3. Create a json file for your custom dataset in src/lib/cfg/. You need to specify the "root" and "train" keys in the json file. You can find some examples in src/lib/cfg/.
4. Add --data_cfg '../src/lib/cfg/your_dataset.json' when training. 

## Acknowledgement
A large part of the code is borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet). Thanks for their wonderful works.

## Citation

```
@article{zhang2021fairmot,
  title={Fairmot: On the fairness of detection and re-identification in multiple object tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={International Journal of Computer Vision},
  volume={129},
  pages={3069--3087},
  year={2021},
  publisher={Springer}
}
```

