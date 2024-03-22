from torchvision import datasets
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def setup_cifar10_data(config):
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(config['seed'])

    if cuda:
        torch.cuda.manual_seed(config['seed'])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=config['train_transforms'])
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=config['test_transforms'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    return train_data,test_data,train_loader, test_loader


train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
    from tqdm import tqdm
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
        
def get_misclassified_images(model, device, test_loader):
    misclassified_images = []
    misclassified_actuals = []
    misclassified_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()
            actual = target
            for i in range(len(pred)):
                if pred[i]!=actual[i]:
                    misclassified_images.append(data[i])
                    misclassified_actuals.append(actual[i])
                    misclassified_predictions.append(pred[i])
    return misclassified_images, misclassified_actuals, misclassified_predictions
    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    
    return test_loss

# Function to plot the training and testing graphs for loss and accuracy
def plt_fig():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot([train_loss.cpu().detach().numpy() for train_loss in train_losses])
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")