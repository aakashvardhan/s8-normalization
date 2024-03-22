from models.model import Net
from models.model_utils import model_summary, sgd_optimizer, save_model, load_model
from config import get_config
from utils import setup_cifar10_data,train,test
import torch
from visualize import show_misclassified_images, plt_misclassified_images
from torch.optim.lr_scheduler import StepLR

def main(config, lr_scheduler=False):
    train_data,test_data,train_loader, test_loader = setup_cifar10_data(config)
    model = Net(config).to(config['device'])
    model_summary(model, input_size=(3, 32, 32))
    optimizer = sgd_optimizer(model, lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=0.1)
    lr = []
    for epoch in range(1,config['epochs']+1):
        print("EPOCH:", epoch)
        train(model, config['device'], train_loader, optimizer, epoch)
        test(model, config['device'], test_loader)
        if lr_scheduler == True:
            scheduler.step()
            lr.append(optimizer.param_groups[0]['lr'])
    
    # format name of model file according to config['norm']
    model_file = 'model_' + config['norm'] + '.pt'
    save_model(model, model_file)
    
    return model, test_loader
    
        
    
