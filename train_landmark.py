import torch
import os
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
    parser.add_argument('--config', type=str, default='model_config.yaml', help='Path to the config file.')
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
    
os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(e) for e in config.GPU)

def main():
    config = load_config()
    os.makedirs(os.path.join(config.LANDMARK_PATH, config.LANDMARK_VERSION), exist_ok=True)
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    
    train_dataset = Dataset(config.config.DATA_TRAIN_GT, config.config.DATA_TRAIN_STRUCTURE, 
                                config.config, config.config.DATA_MASK_FILE)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.config.TRAIN_BATCH_SIZE, 
                                shuffle=True, drop_last=True, num_workers=8)  
    
    model = LandmarkDetectorModel(config.STRUCTURE_LANDMARK_NUM, config.DATA_TRAIN_SIZE)
    model = model.to(device)
    
    for epoch in range(config.LANDMARK_EPOCH):
        for (iter, input) in enumerate(train_loader):
            inputs, _, _, _, landmark = cuda(config, *input)
            
            landmark_gen, loss, logs = model.process(inputs, landmark)
            model.backward(loss)
            
            print('[{}/{}][{}] loss: {}'.format(epoch, config.LANDMARK_EPOCH, iter, loss))
            
        save_image(landmark_gen, os.path.join(config.LANDMARK_PATH, config.LANDMARK_VERSION, 'landmark_epoch-{}_iter-{}.png'.format(epoch, iter)))
        

if __name__ == "__main__":
    main()