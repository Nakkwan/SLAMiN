## Setting
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
2. training dataset 경로 설정의 경우 config.yaml의 아랫쪽에서 설정할 수 있습니다. (defaults: ./datasets/ ~)
3. Ablation을 위한 model1,2를 돌릴지, 전체 모델인 model3를 돌릴지에 대한 설정은 config.yaml의 맨위 MDEOL 변수를 설정함으로써 변경할 수 있습니다. (defaults: 3) 
## Training
```python
python main.py --name=[output name]
```

---
1. Test의 경우 main.py 파일의 load_config함수 안에서 test에 관한 폴더 경로 설정을 할 수 있습니다.
## Test
```python
python test.py --name=[output name]
```

---
1. landmark에 대한 loss를 측정해야하기 때문에 pre-training을 진행했습니다.
## Landmark
```python
python train_landmark.py --name=[output name]
```