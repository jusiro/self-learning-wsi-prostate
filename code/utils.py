########################################################
# Imports
########################################################

import numpy as np
import torch
import os
import cv2
import random
from PIL import Image
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import datetime
from torchvision import models
import pandas as pd
import skimage

random.seed(42)
torch.manual_seed(0)
np.random.seed(0)

########################################################
# Data Generators
########################################################


class Dataset(torch.utils.data.Dataset):

  def __init__(self, dir_images, data_frame, batch_size, classes, input_shape=(3, 224, 224), mode=0,
               data_augmentation=False):

        'Initialization'
        self.dir_images = dir_images
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.classes = classes
        self.data_augmentation = data_augmentation
        self.input_shape = input_shape
        self.mode = mode  # 0: teacher model - global labels, 1: student model - patch level labels,
                          # 2: global labels with global label assignement

        'Secondary Initializations'
        self.images = os.listdir(dir_images)

        if self.mode == 0 or self.mode == 2:  # Slide-level labels

            # Filter patches whose slide is not in the dataframe
            idx = np.in1d([ID.split('_')[0] for ID in self.images], self.data_frame['slide_name'])
            images = [self.images[i] for i in range(self.images.__len__()) if idx[i]]
            self.images = images

            # Filter slides in the dataframe whose patches are not in the images folder
            self.data_frame = self.data_frame[np.in1d(self.data_frame['slide_name'], [ID.split('_')[0] for ID in images])]

        elif self.mode == 1:  # Patch-level labels

            # Filter patches not present in the dataframe
            idx = np.in1d([ID for ID in self.images], self.data_frame['image_name'])
            images = [self.images[i] for i in range(self.images.__len__()) if idx[i]]
            self.images = images

            # Filter patches in the dataframe that are not present in the images folder
            self.data_frame = self.data_frame[np.in1d([ID for ID in self.data_frame['image_name']], [ID for ID in images])]

            # Keep only unique patches in the dataframe (by precaution)
            u, indices = np.unique(self.data_frame['image_name'], return_index=True)
            self.data_frame = self.data_frame[np.in1d(list(range(len(self.data_frame['image_name']))), list(indices))]

        self.indexes = np.arange(len(self.images))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.images[self.indexes[index]]

        # Load image
        x = Image.open(os.path.join(self.dir_images, ID))
        x = np.asarray(x)

        # data augmentation
        if self.data_augmentation:
            x = image_transformation(x)

        # Normalization
        x = norm_image(x, input_shape=self.input_shape)

        # Get label
        if self.mode == 0 or self.mode == 2:
            y = self.data_frame[self.data_frame['slide_name'] == ID.split('_')[0]][self.classes].values
        if self.mode == 1:
            y = self.data_frame[self.data_frame['image_name'] == ID][self.classes].values

        return x, y


class DataGeneratorSlideLevelLabels(object):

    def __init__(self, dataset, batch_size=32, shuffle=False):

        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        'Secondary Initializations'
        self._idx = 0
        self.indexes = np.arange(len(self.dataset.data_frame))
        self._reset()
        self.used_image_names = []
        self.N = 200

    def __len__(self):

        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):

        return self

    def __next__(self):

        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        df_row = self.dataset.data_frame.iloc[self.indexes[self._idx]]

        # Get label
        y = df_row[self.dataset.classes].values[:]

        # Select patches from the slide and load them into a list
        ID = list(df_row[['slide_name']].values)[0]
        images_id = [i for i in range(len(self.dataset.images)) if self.dataset.images[i].split('_')[0] == ID]

        # Memory limitation of patches in one slide
        if len(images_id) > self.N:
            images_id = random.sample(images_id, self.N)
        # Minimum number os patches in a slide (by precaution).
        if len(images_id) < 4:
            images_id.extend(images_id)

        # Load images and include into the batch
        X, Y = [], []
        for i in images_id:
            self.used_image_names.append(self.dataset.images[i])
            x , y = self.dataset.__getitem__(i)
            X.append(x)

            if self.dataset.mode == 0:
                Y = y
            elif self.dataset.mode == 2:
                Y.append(y)

        self._idx += self.batch_size

        return torch.tensor(np.array(X).astype('float32')), torch.tensor(np.array(Y).astype('float32'))

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

########################################################
# Image processing
########################################################


def norm_image(x, input_shape):

    # image resize
    x = cv2.resize(x, (input_shape[1], input_shape[1]))
    # intensity normalization
    x = x / 255.0
    # channel first
    x = np.transpose(x, (2, 0, 1))
    # numeric type
    x.astype('float32')
    return x


def image_transformation(im, border_value=255):

    # Random index for type of transformation
    random_index = np.clip(np.round(random.uniform(0, 1) * 10 / 2), 1, 4)

    if random_index == 1 or random_index == 3:  # translation

        # Randomly obtain translation in pixels in certain bounds
        limit_translation = im.shape[0] / 4
        translation_X = np.round(random.uniform(-limit_translation, limit_translation))
        translation_Y = np.round(random.uniform(-limit_translation, limit_translation))
        # Get transformation function
        T = np.float32([[1, 0, translation_X], [0, 1, translation_Y]])
        # Apply transformation
        im_out = cv2.warpAffine(im, T, (im.shape[0], im.shape[1]), borderValue=(border_value, border_value, border_value))

    elif random_index == 2:  # rotation

        # Get transformation function
        rotation_angle = np.round(random.uniform(0, 360))
        img_center = (im.shape[0] / 2, im.shape[0] / 2)
        T = cv2.getRotationMatrix2D(img_center, rotation_angle, 1)
        # Apply transformation
        im_out = cv2.warpAffine(im, T, (im.shape[0], im.shape[1]), borderValue=(border_value, border_value, border_value))

    elif random_index == 4:  # mirroring

        rows, cols = im.shape[:2]
        src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        dst_points = np.float32([[cols - 1, 0], [0, 0], [cols - 1, rows - 1]])
        T = cv2.getAffineTransform(src_points, dst_points)

        im_out = cv2.warpAffine(im, T, (cols, rows), borderValue=(border_value, border_value, border_value))

    return np.array(im_out)


def extract_patches(im, window_shape=(512, 512, 3), stride=512):
    # Patch extractor function
    patches = skimage.util.view_as_windows(im, window_shape, stride)
    nR, nC, t, H, W, C = patches.shape
    nWindow = nR * nC
    patches = np.reshape(patches, (nWindow, H, W, C))

    return patches


def pad(img, h, w):
    # Zero-padding function
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=255))

########################################################
# Deep Learning
########################################################


def categorical_cross_entropy(y_pred, y_true, class_weights=None, device=None):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)

    if class_weights is None:
        if len(y_true.shape) == 1:
                l = -(y_true * torch.log(y_pred)).mean()
        else:
                l = -(y_true * torch.log(y_pred)).sum(dim=1).mean()
    else:
        class_weights = np.repeat(np.reshape(class_weights, (1, 4)), y_pred.shape[0], axis=0)
        l = -(y_true * torch.log(y_pred) * torch.tensor(class_weights).to(device)).sum(dim=1).mean()
    return l


def patch_level_gg_classifier():

    model = models.vgg16(pretrained=True)

    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.classifier = torch.nn.Sequential(
                       torch.nn.Dropout(.25),
                       torch.nn.Linear(in_features=512, out_features=4),
                       torch.nn.Softmax(dim=1),
    )

    return model


class TeacherModel(torch.nn.Module):

    def __init__(self):
        super(TeacherModel, self).__init__()

        # Backbone
        self.bb = patch_level_gg_classifier()

    def forward(self, images):

        # Patch-Level Classifier
        patch_classification = self.bb(images)

        # Slide-Level max aggregation
        gleason_presence_classification = torch.max(patch_classification, dim=0)[0][1:]

        return gleason_presence_classification


def get_model(mode, device, learning_rate, multi_gpu=False):

    if mode == 0:

        model = TeacherModel()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    elif mode == 1:

        model = patch_level_gg_classifier()
        criterion = categorical_cross_entropy
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model = model.to(device)
    if multi_gpu:
        model = torch.nn.DataParallel(model)

    return model, criterion, optimizer


class CNNTrainer:

    def __init__(self, n_epochs, criterion, optimizer, train_on_gpu, device, dir_out, save_best_only=False,
                 lr_exp_decay=False, multi_gpu=False, learning_rate_esch_half=False, class_weights=False,
                 slice=False):

        'Initialization'
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_on_gpu = train_on_gpu
        self.save_best_only = save_best_only
        self.device = device
        self.dir_out = dir_out
        self.lr_exp_decay = lr_exp_decay
        self.lr0 = optimizer.param_groups[0]['lr']
        self.multi_gpu = multi_gpu
        self.learning_rate_esch_half = learning_rate_esch_half
        self.slice = slice
        self.class_weights = class_weights

        'Secondary Initializations'
        self.history = []
        self.start_time_epoch = []
        self.valid_loss_min = float('inf')

    def train(self, model, train_generator, val_generator=None, epochs_for_weights_update=1):

        for epoch in range(self.n_epochs):

            print('Epoch: {}'.format(epoch + 1) + '/' + str(self.n_epochs))
            self.start_time_epoch = timer()

            # decrease lr after half iterations
            if self.learning_rate_esch_half:
                if epoch == (self.n_epochs / 2) - 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr0 / 10
                    self.lr0 = self.lr0 / 10

            # exponential lr in last epochs
            if self.lr_exp_decay:
                self.optimizer = exp_lr_scheduler(self.optimizer, epoch, lr_decay_epoch=self.n_epochs - 5, lr0=self.lr0)

            # Update model weights
            metric_train, loss_train = self.train_epoch(model, train_generator, is_training=True,
                                                        epochs_for_weights_update=epochs_for_weights_update)

            # Validation performance
            if val_generator is not None:
                print('')
                print('Evaluating validation data...')
                metric_val, loss_val = self.train_epoch(model, val_generator, is_training=False)
            else:
                metric_val, loss_val = [], []

            # Track training progress
            progress_bar(len(train_generator), len(train_generator), metric_train, loss_train,
                         timer() - self.start_time_epoch, metric_val, loss_val)

            # Save best model option
            if self.save_best_only is not False:
                if loss_val < self.valid_loss_min:
                    print('Validation loss improved from ' + str(round(self.valid_loss_min, 5)) + ' to ' + str(
                        round(loss_val, 5)) + '  ... saving model')
                    # Save model
                    if self.multi_gpu is False:
                        torch.save(model.state_dict(), self.dir_out + 'best_model')
                    else:
                        torch.save(model.module.state_dict(), self.dir_out + 'best_model')
                    # Track improvement
                    self.valid_loss_min = loss_val
                else:
                    print('Validation loss did not improve from ' + str(round(self.valid_loss_min, 5)))
            if self.multi_gpu is False:
                torch.save(model.state_dict(), self.dir_out + 'last_model')
            else:
                torch.save(model.module.state_dict(), self.dir_out + 'last_model')

            # Add metrics for history tracking
            self.history.append([loss_train, loss_val, metric_train, metric_val])

        # Load best model if required
        if self.save_best_only is not False:
            model.load_state_dict(torch.load(self.dir_out + 'best_model'))

        # Format history
        history = pd.DataFrame(
            self.history,
            columns=['loss_train', 'loss_val', 'metric_train', 'metric_val'])
        model.eval()

        return model, history

    def train_epoch(self, model, generator, is_training=True, epochs_for_weights_update=1):

        start = timer()
        if is_training:
            # Set to training mode
            model.train()
        else:
            # Set to evaluation mode
            model.eval()

        # keep track of training and validation loss each epoch
        loss_over_all = 0.0
        metric_over_all = 0.0

        # Loop over batches in the generator
        for ii, (data, target) in enumerate(generator):

            if len(data.shape) > 1:

                # Squeeze target
                if len(target.shape)>1:
                    target = target.squeeze(dim=1)

                if is_training:  # Training
                    # Tensors to gpu
                    if self.train_on_gpu:
                        data, target = data.to(self.device), target.to(self.device)

                    # Predicted outputs are log probabilities
                    output = model(data.float())
                    target = target.float()

                    # multi-gpu implementation
                    if self.multi_gpu:

                        output_1 = output[1].view((2, 3))
                        output_1 = torch.max(output_1, dim=0)[0]

                        output_2 = output[2].view((2, 6))
                        output_2 = torch.mean(output_2, dim=0)

                        output = (output[0], output_1, output_2)

                    # Loss and back-propagation of gradients
                    if self.slice:
                        target = target[0, 1:]
                        loss = self.criterion(output, target)
                    else:
                        loss = self.criterion(output, target, self.class_weights, self.device)

                    # Backward propagation
                    loss.backward()

                    if (ii+1) % epochs_for_weights_update == 0:

                        # Update the parameters
                        self.optimizer.step()
                        # Clear gradients
                        self.optimizer.zero_grad()

                else:   # Validation

                    with torch.no_grad():
                        # Tensors to gpu
                        if self.train_on_gpu:
                            data, target = data.to(self.device), target.to(self.device)
                            target = target.float()

                        # Predicted outputs are log probabilities
                        output = model(data.float())

                        # Multi-gpu implementation
                        if self.multi_gpu:
                            output_1 = output[1].view((2, 3))
                            output_1 = torch.max(output_1, dim=0)[0]

                            output_2 = output[2].view((2, 6))
                            output_2 = torch.mean(output_2, dim=0)

                            output = (output[0], output_1, output_2)

                        # Loss and back-propagation of gradients
                        if self.slice:
                            target = target[0, 1:]
                            loss = self.criterion(output, target)
                        else:
                            loss = self.criterion(output, target, self.class_weights, self.device)

                # Track train loss by multiplying average loss by number of examples in batch
                loss_over_all += loss.item()

                # Obtain accuracy
                pred = output >= 0.5
                truth = target >= 0.5
                metric = pred.eq(truth).sum().cpu().data.numpy().item() / target.numel()

                # Multiply average accuracy times the number of examples in batch
                metric_over_all += metric

            # Track training progress
            progress_bar(ii + 1, len(generator), metric, loss.item(), timer() - self.start_time_epoch)

        # Calculate average loss and metrics
        loss_over_all = loss_over_all / len(generator)
        metric_over_all = metric_over_all / len(generator)

        return metric_over_all, loss_over_all


def predict(model, generator, train_on_gpu, device):

    refs = []
    preds = []
    model.eval()

    start_time = timer()
    print('Predicting....')
    # Loop over batches in the generator
    for ii, (data, target) in enumerate(generator):

        # Squeeze target
        if len(target.shape) > 1:
            target = target.squeeze(dim=1)
            # print(target.shape)

        # Tensors to gpu
        if train_on_gpu:
            data, target = data.to(device), target.to(device)

        # Predicted outputs are log probabilities
        out = model(data.float())

        out = out.cpu().detach().data.numpy().tolist()
        target = target.cpu().detach().data.numpy().tolist()

        preds.append(out)
        refs.append(target)

        progress_bar(ii + 1, len(generator), 0, 0, timer() - start_time)

    return np.vstack(preds), np.vstack(refs)

########################################################
# Others
########################################################


def learning_curve_plot(history, dir_out, name_out):

    plt.figure()
    plt.subplot(211)
    plt.plot(history['metric_train'].values)
    plt.plot(history['metric_val'].values)
    plt.axis([0, history['loss_train'].values.shape[0]-1, 0, 1])
    plt.legend(['acc', 'val_acc'], loc='upper right')
    plt.title('learning-curve')
    plt.ylabel('accuracy')
    plt.subplot(212)
    plt.plot(history['loss_train'])
    plt.plot(history['loss_val'])
    plt.axis([0, history['loss_train'].values.shape[0]-1, 0, 1])
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(dir_out + '/' + name_out)
    plt.close()


def progress_bar(batch, total_batches, metric, loss, eta, metric_val='', loss_val=''):

    n = 40
    batches_per_step = max(total_batches // n, 1)
    eta = str(datetime.timedelta(seconds=eta))[2:7]

    bar = 'Batch ' + str(batch) + '/' + str(total_batches) + ' -- ['

    for i in range((batch // batches_per_step) + 1):
        bar = bar + '='
    bar = bar + '>'
    for ii in range(n - (batch // batches_per_step + 1)):
        bar = bar + '.'

    if metric_val == '':
        bar = bar + '] -- metric: ' + str(round(metric, 4)) + ' -- loss: ' + str(round(loss, 4)) + ' -- ETA: ' + eta
        print(bar, end='\r')
    else:
        bar = bar + '] -- metric: ' + str(round(metric, 4)) + ' -- loss: ' + str(
            round(loss, 4)) + ' -- val_metric: ' + str(round(metric_val, 4)) + ' -- val_loss: ' + str(
            round(loss_val, 4)) + ' -- ETA: ' + eta
        print(bar)


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=100, lr0=0.001):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch < lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr0 * np.exp(-0.25 * (epoch - lr_decay_epoch))

    return optimizer
