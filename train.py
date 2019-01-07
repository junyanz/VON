# import torch.backends.cudnn as cudnn
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
dataset = create_dataset(opt)
dataset_size = len(dataset)
print('#training data = %d' % dataset_size)
model = create_model(opt)
model.setup(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    save_result = True
    iter_data_time = time.time()

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - dataset_size * (epoch - opt.epoch_count)
        model.set_input(data)
        if model.skip():
            continue
        model.update_G()
        model.update_D()
        model.check_nan_inf()

        if save_result or total_steps % opt.display_freq == 0:
            save_result = save_result or total_steps % opt.update_html_freq == 0
            if model.visual_names:
                visualizer.display_current_results(model.get_current_visuals(), epoch, ncols=2, save_result=save_result)
            save_result = False

        if total_steps % opt.print_freq == 0:
            losses = model.get_current_losses()
            t_model = time.time() - iter_start_time
            t_data = iter_start_time - iter_data_time
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_model, t_data)
            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            model.clear_running_mean()

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save_networks('latest')

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()
