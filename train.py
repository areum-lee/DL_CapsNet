from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm
import torchnet as tnt

model = CapsuleNet()
# model.load_state_dict(torch.load('epochs/epoch_327.pt'))
model.cuda()

print("# parameters:", sum(param.numel() for param in model.parameters()))

optimizer = Adam(model.parameters())

engine = Engine()
meter_loss = tnt.meter.AverageValueMeter()
meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

train_loss_logger = VisdomPlotLogger('line', opts={'title': '120160_layer Train Loss'})
train_error_logger = VisdomPlotLogger('line', opts={'title': '120160_layer Train Accuracy'})
test_loss_logger = VisdomPlotLogger('line', opts={'title': '120160_layer Test Loss'})
test_accuracy_logger = VisdomPlotLogger('line', opts={'title': '120160_layer Test Accuracy'})
confusion_logger = VisdomLogger('heatmap', opts={'title': '120160_layer Confusion matrix',
                                                 'columnnames': list(range(NUM_CLASSES)),
                                                 'rownames': list(range(NUM_CLASSES))})
ground_truth_logger = VisdomLogger('image', opts={'title': '120160_layer Ground Truth'})
reconstruction_logger = VisdomLogger('image', opts={'title': '120160_layer Reconstruction'})

capsule_loss = CapsuleLoss()


def get_iterator(mode):
    
    #dataset = MNIST(root='./data', download=True, train=mode)
    #data = getattr(dataset, 'train_data' if mode else 'test_data')
    #labels = getattr(dataset, 'train_labels' if mode else 'test_labels')

    d,l =  next(iter(dset_loaders['train']))
    tensor_dataset = tnt.dataset.TensorDataset([data, labels])

    return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)


def processor(sample):
    data, labels, training = sample

    data = augmentation(data.unsqueeze(1).float() / 255.0)
    labels = torch.LongTensor(labels)

    labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

    data = Variable(data).cuda()
    labels = Variable(labels).cuda()

    if training:
        classes, reconstructions = model(data, labels)
    else:
        classes, reconstructions = model(data)

    loss = capsule_loss(data, labels, classes, reconstructions)

    return loss, classes


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


def on_sample(state):
    state['sample'].append(state['train'])


def on_forward(state):
    meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    #print(state['output'].data.size())
    #print(torch.LongTensor(state['sample'][1]).size())
    one_hot = torch.eye(NUM_CLASSES)[torch.LongTensor(state['sample'][1])].type(torch.LongTensor).view(1,-1)
    #print('onehot',one_hot)
    #confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    confusion_meter.add(state['output'].data, one_hot)
    
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

    reset_meters()

    #engine.test(processor, get_iterator(False))
    engine.test(processor, dset_loaders['val'])    

    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    confusion_logger.log(confusion_meter.value())

    print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    torch.save(model.state_dict(), '/data2/ar/gestures20170903/epoch/120160_layer/epoch_%d.pt' % state['epoch'])

    # Reconstruction visualization.

    test_sample = next(iter(dset_loaders['val']))

    ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
    _, reconstructions = model(Variable(ground_truth).cuda())
    reconstruction = reconstructions.cpu().view_as(ground_truth).data

    ground_truth_logger.log(
        make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
    reconstruction_logger.log(
        make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

# def on_start(state):
#     state['epoch'] = 327
#
# engine.hooks['on_start'] = on_start
engine.hooks['on_sample'] = on_sample
engine.hooks['on_forward'] = on_forward
engine.hooks['on_start_epoch'] = on_start_epoch
engine.hooks['on_end_epoch'] = on_end_epoch

engine.train(processor, dset_loaders['train'], maxepoch=NUM_EPOCHS, optimizer=optimizer)