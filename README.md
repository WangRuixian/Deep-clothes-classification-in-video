Author: Ethan,

Email: 976264378@qq.com

Current Affiliation: UESTC

Future Affiliation:  NTU

supervisor: Jin Qi

Date: 8/6/2022

note: this work was done while I pursued my bachelor/master degree in Jin Qi's AIML Lab in UESTC

#Deep clothes wearing classification in video

### Installation
```bash
pip install -r requirements.txt
```

### Reproduce results
1. Run `main.py` to produce final classification result
2. Run `cloth_detection.py` to produce images with clothes detection bounding boxes.

### Model weights
We trained a Yolo-v3 object detection on [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset, pre-trained model weights (tensorflow weights and darknet weights) can be download [here](https://drive.google.com/file/d/1DPydA0FpLYEHaFYDa8_oZAot_Ou5JefK/).

### Dataset
For the classifier, we use a relatively small dataset consists of only 46 clothes of 2 classes (clothes with stripes and clothes without stripes), the dataset can be download [here](https://drive.google.com/file/d/1oCMPB1MSsB3yJdOLm2iEZFGyYSKXQmIw/). 
![](./images/clothes_class.jpg)

## System pipeline
![](./images/system_pipeline.png)

## References
1. Yannis Kalantidis, Lyndon Kennedy & Li-Jia Li. (2013) "Getting the Look: Clothing Recognition and Segmentation for Automatic Product Suggestions in Everyday Photos".

2. Yuying Ge, Ruimao Zhang, Lingyun Wu, Xiaogang Wang, Xiaoou Tang & Ping Luo. (2019) "DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images".

3. Joseph Redmon & Ali Farhadi. (2018) "YOLOv3: An Incremental Improvement".