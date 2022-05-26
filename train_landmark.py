import torch
import os
import numpy as np
import cv2
from src.data import Dataset
from src.network import LandmarkDetectorModel
from torch.utils.data import DataLoader
import argparse
from src.config import Config
from torchvision.utils import save_image, make_grid

def load_config():
    r"""loads model config 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='output model name.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    parser.add_argument('--path', type=str, default='./results', help='outputs path')
    parser.add_argument("--resume_all", action="store_true", help='load model from checkpoints')
    parser.add_argument("--remove_log", action="store_true", help='remove previous tensorboard log files')
    
    opts = parser.parse_args()
    config = Config(opts, 'train')
    output_dir = os.path.join(opts.path, opts.name)
    perpare_sub_floder(output_dir)
        
    return config

def perpare_sub_floder(output_path):
    img_dir = os.path.join(output_path, 'images')
    if not os.path.exists(img_dir):
        print("Creating directory: {}".format(img_dir))
        os.makedirs(img_dir)     


    checkpoints_dir = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        print("Creating directory: {}".format(checkpoints_dir))
        os.makedirs(checkpoints_dir) 
        
def cuda(config, *args):
        return (item.to(config.DEVICE) for item in args)
    
def generate_landmark_map(landmark_cord, img_size = 256):
        '''
        :param landmark_cord: [B,self.config.LANDMARK_POINTS,2] or [self.config.LANDMARK_POINTS,2], tensor or numpy array
        :param img_size:
        :return: landmark_img [B, 1, img_size, img_size] or [1, img_size, img_size], tensor or numpy array
        '''

        if len(landmark_cord.shape) == 3:
            landmark_img = np.zeros((landmark_cord.shape[0],3,img_size, img_size), dtype="uint8")
            for i in range(landmark_cord.shape[0]):
                img = np.zeros((img_size, img_size, 3), dtype="uint8")
                for j in range(len(landmark_cord[i,:,0])):
                    cv2.line(img, (landmark_cord[i,j,0], landmark_cord[i,j,1]), (landmark_cord[i,j,0], landmark_cord[i,j,1]), (255, 0, 0), 5)
                img = np.moveaxis(img, -1, 0)
                landmark_img[i] = img
        elif len(landmark_cord.shape) == 2:
            landmark_img = np.zeros((1,3,img_size, img_size), dtype="uint8")
            img = np.zeros((img_size, img_size, 3), dtype="uint8")
            for i in range(len(landmark_cord)):
                cv2.line(img, (landmark_cord[i,0], landmark_cord[i,1]), (landmark_cord[i,0], landmark_cord[i,1]), (255, 0, 0), 5)
            img = np.moveaxis(img, -1, 0)
            landmark_img[0] = img

        return torch.from_numpy(landmark_img).type(torch.FloatTensor)

    
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def main():
    config = load_config()
    os.makedirs(os.path.join(config.LANDMARK_PATH, config.LANDMARK_VERSION, 'checkpoints'), exist_ok=True)
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")
        
    
    train_dataset = Dataset(config.DATA_TRAIN_GT, config.DATA_TRAIN_STRUCTURE, 
                                config, landmark_file=config.DATA_TRAIN_LANDMARK)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.TRAIN_BATCH_SIZE, 
                                shuffle=True, drop_last=True, num_workers=8)  
    
    model = LandmarkDetectorModel(config, config.STRUCTURE_LANDMARK_NUM, config.DATA_TRAIN_SIZE)
    model = model.to(config.DEVICE)
    
    for epoch in range(config.LANDMARK_EPOCH):
        for (iter, input) in enumerate(train_loader):
            inputs, _, _, _, landmark = cuda(config, *input)
            
            landmark_gen, loss, logs = model.process(inputs, landmark)
            model.backward(loss)
            
            print('[{}/{}][{}] loss: {}'.format(epoch, config.LANDMARK_EPOCH, iter, loss))
        
        landmark_img = generate_landmark_map(np.array(landmark_gen.detach().cpu().type(torch.int32)))
       
        save_image(landmark_img, os.path.join(config.LANDMARK_PATH, config.LANDMARK_VERSION, 'landmark_epoch-{}_iter-{}.png'.format(epoch, iter)))
        save_image(inputs, os.path.join(config.LANDMARK_PATH, config.LANDMARK_VERSION, 'landmark_epoch-{}_iter-{}_inputs.png'.format(epoch, iter)))
            
        if epoch % 10 == 0:
            model.save(os.path.join(config.LANDMARK_PATH, config.LANDMARK_VERSION, 'checkpoints', 'landmark_{}.pth'.format(epoch)))
        

if __name__ == "__main__":
    main()