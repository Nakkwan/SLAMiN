# SLAMiN 
    - UNIST DGMS Final Project Team2
    - Code is implemented based on [structureflow](https://github.com/RenYurui/StructureFlow)
 
## Inpainting Results Example

## Setting
- Environment for RTX 3090
    - pytorch 1.8
    - CUDA 11.1
- Package requirements
    - Pandas
    - Tensorbaord
    - PIL
    - cv2
    - pyyaml

flow structure부분에서 flow를 흘려주는 부분에 cuda package에 대한 부분을 설정해주어야합니다. 
현재 돌리는 세팅의 경우 RTX 3090, pytorch 1.8, python 3.8, CUDA 11.1 입니다.
GPU와 환경마다 맞는 설정이 다르기 때문에
```python 
cd ./resample2d_package
python setup.py install --user
```
을 통해서 해당 package를 깔아줄 때에 gpu의 architecture에 맞게 nvcc_args를 변경해주어야합니다. 
대부분의 환경에서 70, 75, 86 중 하나로 설정하면 진행이 가능했습니다.

1. 훈련에 관한 모든 변수 및 설정들은 config.yaml에 들어있습니다
2. Ablation을 위한 model1,2를 돌릴지, 전체 모델인 model3를 돌릴지에 대한 설정은 config.yaml의 맨위 MODEL 변수를 설정함으로써 변경할 수 있습니다. (defaults: 3) 
## Training
1. training dataset 경로 설정의 경우 config.yaml의 아랫쪽에서 설정할 수 있습니다. (defaults: ./datasets/ ~)
    ```
    DATA_TRAIN_GT: datasets/img_align_celeba/img_align_celeba
    DATA_TRAIN_STRUCTURE: datasets/img_align_structure
    DATA_TRAIN_LANDMARK: datasets/list_landmarks_align_celeba.csv
    DATA_VAL_GT: datasets/validation
    DATA_VAL_STRUCTURE: datasets/validation_structure
    DATA_VAL_MASK:
    ```
2. Landmark에 관한 csv파일의 경우 [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?select=list_landmarks_align_celeba.csv)의 landmark를 사용했습니다.
2. name의 경우 results와 checkpoint에 저장될 폴더의 이름입니다. (defaults: team2)
```python
python main.py --name=[output name]
```

---
## Test
1. Test의 경우 main.py 파일의 load_config함수 안에서 test에 관한 폴더 경로 설정을 할 수 있습니다.
    - defaults로는 datasets/test/input과 datasets/test/masks로 설정되어 있습니다. (수업에서 제공된 test dataset 600장)
2. name의 경우 checkpoint에서 load될 가중치 폴더의 이름입니다. (defaults: team2)
```python
python test.py --name=[output name]
```

---
1. landmark에 대한 loss를 측정해야하기 때문에 pre-training을 진행했습니다.
## Landmark
```python
python train_landmark.py --name=[output name]
```