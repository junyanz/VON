import os
from os.path import join
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images, convert_image
from util.util_voxel import save_vox_to_obj, render
from util import html
from tqdm import tqdm

# options
opt = TestOptions().parse()
opt.num_threads = 1
opt.serial_batches = True  # no shuffle

# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.setup(opt)
model.netG_3D.eval()
model.set_posepool(dataset.dataset.datasets[0].get_posepool())
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, '{:s}_views{}_shape{}_r{}'.format(opt.name, opt.n_views, opt.n_shapes, opt.random_view))
webpage = html.HTML(web_dir, 'Training = %s, %s, 2D = %s, 3D = %s' % (opt.name, opt.phase, opt.model2D_dir, opt.model3D_dir))
model_path = os.path.join(web_dir, 'images')
count = 0

prog_bar = tqdm(total=opt.n_shapes)

while (True):
    if count == opt.n_shapes:
        break
    for n, data in enumerate(dataset):
        if count == opt.n_shapes:
            break
        count += 1

        # if opt.real_shape or opt.real_texture:
        model.reset_shape(opt.reset_shape and not opt.real_shape)
        model.reset_texture(opt.reset_texture and not opt.real_texture)
        model.set_input(data, opt.reset_shape and opt.real_shape, opt.reset_texture and opt.real_texture)
        # model.eval_rec()
        if not opt.real_shape:
            model.sample_3d()
        all_images, all_names = [], []
        if opt.render_25d:
            all_depths, all_depth_names = [], []
            all_masks, all_mask_names = [], []

        if opt.show_input:
            input_real = convert_image(model.input_B)
            all_images.append(input_real)
            all_names.append('real')
        if opt.show_rec:
            rec_B = convert_image(model.rec_B)
            all_images.append(rec_B)
            all_names.append('rec')
        for k in range(opt.n_views):
            model.reset_view(reset=True)
            image, depth, mask = model.sample_2d(view_id=k, extra=True)
            image_np = convert_image(image)
            all_images.append(image_np)
            all_names.append('view_{:03d}'.format(k))
            if opt.render_25d:
                depth_np = convert_image(depth)
                mask_np = convert_image(mask)
                all_depths.append(depth_np)
                all_masks.append(mask_np)
                all_depth_names.append('depth_{:03d}'.format(k))
                all_mask_names.append('mask_{:03d}'.format(k))

        if opt.render_3d:
            obj_name = join(model_path, 'shape%03d.obj' % (count))
            save_vox_to_obj(model.voxel.data.cpu().numpy(), 0.5 if not opt.use_df else 0.85, obj_name)
            render_prefix = join(model_path, 'shape{:03d}'.format(count))
            render(obj_name, model.views, render_prefix, 512)

        img_path = 'shape{:03d}'.format(count)
        model.count += 1
        save_images(webpage, all_images, all_names, img_path, None,
                    width=opt.fine_size, aspect_ratio=opt.aspect_ratio)
        if opt.render_25d:
            save_images(webpage, all_depths, all_depth_names, img_path, None,
                        width=opt.fine_size, aspect_ratio=opt.aspect_ratio)
            save_images(webpage, all_masks, all_mask_names, img_path, None,
                        width=opt.fine_size, aspect_ratio=opt.aspect_ratio)
        webpage.save()
        prog_bar.update(1)

    webpage.save()
