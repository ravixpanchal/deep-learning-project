import argparse
import os
import numpy as np
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from model import resnet50_official
from torchsummary import summary

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from sklearn.metrics import accuracy_score

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--s', type=float, default=0,
                    help='scale sparse rate (default: 0)')
parser.add_argument('--save', default='./trainAll_prunued/', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--refine', default='model_best_fine.pth.tar', type=str, metavar='PATH',
                    help='the PATH to pruned model')

device = torch.device("cuda:0" if torch.cuda.is_available() and not False else "cpu")
print("DEVICE", device)

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    if args.refine:
        checkpoint = torch.load(args.refine)
        #cfg = [64, 64, 64, 256, 48, 48, 256, 48, 48, 256, 96, 96, 512, 96, 96, 512, 96, 96, 512, 64, 64, 512, 
        #        192, 192, 1024, 192, 192, 1024, 192, 192, 1024, 128, 128, 1024, 128, 128, 1024, 192, 192, 1024, 384,
        #        394, 2048, 384, 384, 2048, 512, 512] # 48 layers

        #cfg=[64, 15, 6, 128, 14, 8, 128, 256, 71, 46, 47, 51, 256, 512, 512, 512] # pr config,fllters left
        model = resnet50_official()
        #print("CFG=", cfg)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    if args.refine:
        model.load_state_dict(checkpoint['state_dict'])
        best_prec1 = checkpoint['best_prec1']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.refine, checkpoint['best_prec1']))
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print("Number of para in Original Model", num_parameters)
        print("pruned model without change in last layer")
        summary(model, (3, 224,224))
    else:
        print("=> LOAD ERROR no checkpoint found at '{}'".format(args.refine))

    #exit()
    # define loss function (criterion) and optimizer

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join('', '/storage/research/data/Tejalal/txtfiles/B_cancer/data30/train')
    valdir = os.path.join('', '/storage/research/data/Tejalal/txtfiles/B_cancer/data30/val')

    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
        	transforms.Resize(200),
        	transforms.CenterCrop(200),
            #transforms.RandomResizedCrop(50),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(200),
            transforms.CenterCrop(200),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    print("\n\n\n after FC change")
    num_ftrs = model.module.fc.in_features
    model.module.fc = nn.Linear(num_ftrs, 2)
    newmodel=model.cuda()
    summary(newmodel, (3, 200, 200))
    #exit()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(newmodel.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #history_score = np.zeros((args.epochs + 1, 1))
    #np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        #train1, train5, train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        #prec1, test5, test_loss = validate(val_loader, model, criterion)
        #history_score[epoch] = prec1
        #np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')

        top1acc, newmodel = train(newmodel, epoch, train_loader, optimizer)
        prec1, actuals, probabilities = test(newmodel, val_loader)
        prec1=prec1.item()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': newmodel.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save, actuals, probabilities, newmodel, val_loader, epoch+1)

        #with open("finietued_model_log.txt", 'a+') as textfile:
        #	textfile.write('\n {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(train1, train5, train_loss, prec1, test5, test_loss))

        with open(args.save+"resnet34_finetunetxt_trainALL.txt", 'a+') as textfile:
            textfile.write('\n {:.2f} {:.2f}'.format(top1acc, prec1))

        print("\n\n\n--------EPOCH DATA-----------")
        getMetrices(newmodel, val_loader, fname='Last_') # print metrics for every epochs
        which_class = 0 # Healthy, NON IDC
        actuals, class_probabilities = test_class_probabilities(newmodel, device, val_loader, which_class)
        plotROC(actuals, class_probabilities, fname='Last_') # ROC curve
        #print("ROC EXECUTED")
        plotPRcurve(actuals, class_probabilities, fname='Last_') # PR Curve

    #history_score[-1] = best_prec1
    #np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')


def train(model, epoch, train_loader, optimizer):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    acc = train_acc / float(len(train_loader.dataset)) * 100.0
    
    return acc, model

'''
def testold(model, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    print("CORRECT=", correct)
    return correct / float(len(val_loader.dataset)) * 100.0
'''

def test(model, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    actuals = [] # plot roc
    probabilities = [] # plot roc
    for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # plot roc, get probalities
        which_class=0 # healthy class
        prediction = output.argmax(dim=1, keepdim=True)
        actuals.extend(target.view_as(prediction) == which_class)
        output=output.detach().cpu().numpy() # detach
        probabilities.extend(np.exp(output[:, which_class]))

    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    print("CORRECT=", correct)
    return correct / float(len(val_loader.dataset)) * 100.0, [i.item() for i in actuals], [i.item() for i in probabilities]

def save_checkpoint(state, is_best, filepath, actuals, probabilities, newmodel, val_loader, epochn, name='finetunedTL_checkpoint_avg75.pth.tar'):
    torch.save(state, os.path.join(filepath, name))
    if is_best:
        shutil.copyfile(os.path.join(filepath, name), os.path.join(filepath, 'finetunedTL_trainAll.pth.tar'))
        print("\n\n BEST MODEL METRICS, Epochs number=", epochn)

        getMetrices(newmodel, val_loader, fname='Best_')
        which_class = 0 # Healthy, NON IDC
        actuals, class_probabilities = test_class_probabilities(newmodel, device, val_loader, which_class)
        plotROC(actuals, class_probabilities, fname='Best_') # ROC curve
        print("ROC EXECUTED")
        plotPRcurve(actuals, class_probabilities, fname='Best_') # PR Curve

def getMetrices(newmodel, val_loader, fname):
    nb_classes = 2
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(val_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = newmodel(inputs)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

    # Confusion matrix
    cm1=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print("Confiusion matrix ", cm1)
    total1=sum(sum(cm1))

    print("total samples= ", total1)

    #####from confusion matrix calculate accuracy
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy by CM= ', accuracy1)

    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1]) # TP/(TP+FP) 
    sensitivity2 = cm1[0,0]/(cm1[0,0]+cm1[1,0]) # TP/(TP+FN), ,recall
    print('Sensitivity : ', sensitivity1, sensitivity2)

    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    specificity2 = cm1[1,1]/(cm1[0,1]+cm1[1,1]) # TN/(TN+FP)
    print('Specificity : ', specificity1, specificity2)

    # Accuracy

    print("ACC by ACC_SCORE", accuracy_score(lbllist.numpy(), predlist.numpy())) # Y_test, Y_pred

    target_names= ['0', '1']
    print(classification_report(lbllist.numpy(), predlist.numpy(), target_names=target_names))

    #average_precision = average_precision_score(y_test, y_score)
    #print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    conf_mat=cm1
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print("Per class acuracy", class_accuracy)

    X_test = predlist.numpy()
    y_test = lbllist.numpy()

    plot_confusion_matrix(cm=cm1, 
                      normalize    = False,
                      target_names = ['0', '1'],
                      title        = "Confusion Matrix",
                      cmap='Blues', fname=fname+'ConfusionMatrix')

def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            #print("TARGETS", target)
            prediction = output.argmax(dim=1, keepdim=True)
            #print("PREDICTION", prediction)
            actuals.extend(target.view_as(prediction) == which_class)
            output=output.cpu().numpy()
            probabilities.extend(np.exp(output[:, which_class]))
    return [i.item() for i in actuals], [i.item() for i in probabilities]

def plotROC(Y_test1, rf_probs, fname):
    #rf_probs=p
    #print("Y_test1", Y_test1)
    #print("\n\nRF ", rf_probs)
    roc_value = roc_auc_score(Y_test1, rf_probs)
    print('ROC Value = ',roc_value )
    base_fpr, base_tpr, _ = roc_curve(Y_test1, [1 for _ in range(len(Y_test1))])
    model_fpr, model_tpr, _ = roc_curve(Y_test1, rf_probs)
    #print("FPR=", model_fpr, "TPR=", model_tpr)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16  
    name='ROC curve'
    # Plot both curves
    plt.plot(base_fpr, base_tpr, color='b',  label = 'baseline', linestyle='--')
    plt.plot(model_fpr, model_tpr, color='orange', label=name+'(area = %0.2f)' % roc_value)
    plt.legend();
    plt.xlabel('False Positive Rate (FPR)'); 
    plt.ylabel('True Positive Rate (TPR)'); 
    #plt.title('ROC Curves');
    plt.savefig(args.save+fname+'ROC_curve.png', dpi=400)
    plt.close

def plotPRcurve(Y_test1, rf_probs, fname):
    precision, recall, thresholds = precision_recall_curve(Y_test1, rf_probs)
    #f1  = f1_score(Y_test1, rf_probs)
    aucs = auc(recall, precision)
    print("AUC=", auc)

    base_fpr, base_tpr, _ = roc_curve(Y_test1, [1 for _ in range(len(Y_test1))])

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16  
    name='Precision-Recall curve'
    # Plot both curves
    #plt.plot(base_fpr, base_tpr, color='b',  label = 'baseline', linestyle='--')
    plt.plot(recall, precision, lw=2, color='b', label=name+'(area = %0.2f)' % aucs)
    plt.legend();
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(args.save+fname+'PR_curve.png', dpi=400)

def plot_confusion_matrix(cm, target_names, fname, title='Confusion matrix', cmap=None, normalize=True):
    """
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues, inferno, magma, Oranges, OrRd
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    #cm = np.array([[4, 1],
    #               [1, 2]])
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('True label')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig(args.save+fname+'.png', dpi=400)


if __name__ == '__main__':
    main()