from options.test_options import TestOptions
from models import create_model
from os.path import join
from skimage import measure
import numpy as np
import os
from tqdm import tqdm
import torch
from util.util_voxel import render


def save_obj(vertices, faces, name='model'):
    with open(name + '.obj', 'w') as f:
        for v in vertices:
            f.write('v {} {} {}\n'.format(v[0] - 0.5, v[1] - 0.5, v[2] - 0.5))
        for face in faces:
            f.write('f {} {} {}\n'.format(
                face[0] + 1, face[1] + 1, face[2] + 1))


opt = TestOptions().parse()
opt.df_sigma = 8.0
opt.use_df = opt.use_df or 'df' in opt.checkpoints_dir
model = create_model(opt)
# model.setup(opt)
model.eval()
net_path = join(opt.checkpoints_dir, '%s_net_G_3D.pth' % opt.epoch)
net_dict = torch.load(net_path)
model.netG_3D.module.load_state_dict(net_dict)

sampled_shapes, interp_traj = model.sample(k=opt.n_shapes, interp_traj=2 if opt.interp_shape else 0, step=10)
result_root = join(opt.checkpoints_dir, 'test_epoch_%s' % opt.epoch)
os.makedirs(result_root, exist_ok=True)
shape_dir = join(result_root, 'sampled_shapes')
os.makedirs(shape_dir, exist_ok=True)
np.savez(join(shape_dir, 'sample_shapes'), df=sampled_shapes)
space = 1.0 / float(opt.voxel_res)
if not opt.use_df:
    opt.ios_th = 0.5
print('thresholding = %f' % opt.ios_th)
views = np.zeros([6, 2])
views[:, 0] = 30
views[:, 1] = np.linspace(0, 360, 6, endpoint=False)
for idx, s in enumerate(tqdm(sampled_shapes)):
    output = -np.log(s) / opt.df_sigma if opt.use_df else s
    v, f, n, _ = measure.marching_cubes_lewiner(output, opt.ios_th, spacing=(space, space, space))
    save_obj(v, f, join(shape_dir, '%04d' % idx))
    if opt.render_3d:
        render(join(shape_dir, '%04d.obj' % idx), views, '%04d' % idx, 512)

if opt.interp_shape:
    traj_dir = join(result_root, 'sampled_interpolation')
    os.makedirs(traj_dir, exist_ok=True)
    for idx, traj in enumerate(tqdm(interp_traj)):
        save_dir = join(traj_dir, 'traj_%04d' % idx)
        os.makedirs(save_dir, exist_ok=True)
        for step, s in enumerate(traj):
            output = -np.log(s) / opt.df_sigma if opt.use_df else s
            v, f, n, _ = measure.marching_cubes_lewiner(output, opt.ios_th, spacing=(space, space, space))
            save_obj(v, f, join(save_dir, 'step_%04d' % step))
            if opt.render_3d:
                render(join(shape_dir, 'step_%04d.obj' % idx), views, 'step_%04d' % idx, 512)
