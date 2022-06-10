1. python main.py --name=[저장될 파일, 실험이랑] --path=[결과 파일 경로 (defaults:results)] 
     
2. config.yaml파일 아래에 각 파일 경로 설정
3. 맨 아래 landmark의 경우 landmark만 따로 돌릴 시에 설정해주면 됩니다
4. config의 STRUCTURE_L1_WEIGHT과 landmark관련 변수들은 추가했는데 기본 L1의 weight가 4로 되어있어서, 일단 blur 넣은거랑 기본이랑 반반해서 2씩 줬습니다
    - 그냥 blur만 하면 모서리 부분은 아예 안들어가서 기본 L1이랑 합쳤습니다
    - blur weight는 model.py 파일에서 조절하시면 됩니다
    - 자잘한거 말고 추가한 큰 블럭들은 ####### fix ######### 같은 칸 안에 넣었습니다
    - blur weight의 정도는 mask3이라고 되어있는 이미지에 임의로 뽑아봤습니다. 

5. validation의 경우 저희에게 교수님이 주신 16285개 들어있는 train dataset으로 생각하고 짰습니다
    - 이 dataset의 경우 landmark 데이터가 없어서 forward만 하는 validation에 넣었습니다.

6. structure model부분만 건드렸기때문에 model 2, 3의 경우는 확인을 아직 안해봤습니다.



---
1. l1_init
    1. weight 4, 0 