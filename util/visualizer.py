import numpy as np
import os
import ntpath
from . import util
from . import html
from scipy.misc import imresize
import math
import torch
from subprocess import Popen, PIPE
import sys
import socket


def convert_image(input_image):
    if torch.is_tensor(input_image):
        if input_image.requires_grad:
            image = util.tensor2im(input_image.data)
        else:
            image = util.tensor2im(input_image)
    else:
        image = input_image
    return image


def convert_error(input_error):
    error = input_error
    if torch.is_tensor(input_error):
        error = input_error.mean()
        if getattr(error, 'is_cuda'):
            error = error.cpu()
        if error.requires_grad:
            error = error.item()
    else:
        error = input_error
    return error


def save_images(webpage, images, names, image_path, title=None, width=256, aspect_ratio=1.0):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path)
    # name = os.path.splitext(short_path)[0]
    name = short_path
    if not title:
        title = name
    webpage.add_header(title)
    ims = []
    txts = []
    links = []

    for label, im in zip(names, images):
        image_name = '%s_%s.jpg' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.log_path = os.path.join(
            opt.checkpoints_dir, opt.name, 'train_log.txt')
        self.prefix = socket.gethostname() + ' gpu %s' % (str(opt.gpu_ids))
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)
            if not self.vis.check_connection():
                self.throw_visdom_connection_error()

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            util.mkdirs([self.web_dir, self.img_dir])

    # |visuals|: dictionary of images to display or save

    def throw_visdom_connection_error(self):
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, ncols=2, save_result=False, image_format='jpg'):
        if self.display_id > 0:  # show images in the browser
            title = self.name
            nrows = int(math.ceil(len(visuals.items()) / float(ncols)))
            images = []
            idx = 0
            for label, image in visuals.items():
                image_numpy = convert_image(image)
                title += " | " if idx % nrows == 0 else ", "
                title += label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
            if len(visuals.items()) % ncols != 0:
                white_image = np.ones_like(
                    image_numpy.transpose([2, 0, 1])) * 255
                images.append(white_image)
            try:
                self.vis.images(images, nrow=nrows, win=self.display_id + 1,
                                opts=dict(title=title))
            except IOError:
                self.throw_visdom_connection_error()
                self.vis.images(images, nrow=nrows, win=self.display_id + 1,
                                opts=dict(title=title))

        if self.use_html and save_result:  # save images to a html file
            for label, image in visuals.items():
                image_numpy = convert_image(image)
                img_path = os.path.join(
                    self.img_dir, 'epoch%.3d_%s.%s' % (epoch, label, image_format))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []
                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.%s' % (n, label, image_format)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if self.display_id <= 0:
            return
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append(
            [convert_error(losses[k]) for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] *
                           len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except IOError:
            self.throw_visdom_connection_error()
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] *
                           len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)

    # losses: same format as |losses| of plotCurrentlosses
    def print_current_losses(self, epoch, i, losses, t_model, t_data):
        message = '%s:  (epoch: %d, iters: %d, model: %.3f, data: %.3f) ' % (self.prefix, epoch, i, t_model, t_data)
        for k, v in losses.items():
            message += ', %s: %.3f' % (k, v)

        print(message)
        # write losses to text file as well
        with open(self.log_path, "a") as log_file:
            log_file.write(message + '\n')
