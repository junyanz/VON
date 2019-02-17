import numpy as np
import numba
from scipy.interpolate import RegularGridInterpolator as rgi
from skimage import measure
from subprocess import call
from os.path import dirname, join
import cv2

try:
    from .util_print import str_warning
except ImportError:
    str_warning = '[Warning]'
from glob import glob


def render(obj_name, views, render_prefix, res):
    render_script = join(dirname(__file__), 'util_render.py')
    cmd = 'blender --background --python %s %s %s' % (render_script, obj_name, render_prefix)
    for view_id in range(views.shape[0]):
        cmd_view = cmd + ' %f %f %d >/dev/null' % (views[view_id, 0], views[view_id, 1], view_id)
        call(cmd_view, shell=True)
    imgs = glob(render_prefix + '*.png')
    crop_and_pad(imgs)


def crop_and_pad(img_paths):
    for img_path in img_paths:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            mask = img[:, :, 3] / 255
            mask = mask[:, :, np.newaxis]
            bg = np.ones([img.shape[0], img.shape[1], 3]) * 255
            img = img[:, :, :-1] * mask + bg * (1 - mask)
            idh, idw, _ = np.where(mask > 0.8)
            idh_min, idh_max, idw_min, idw_max = [np.min(idh), np.max(idh), np.min(idw), np.max(idw)]
            img = img[idh_min:idh_max, idw_min:idw_max]
            cropped = __pad_real_im(img).astype(np.uint8)
            cv2.imwrite(img_path, cropped)
        except IOError:
            continue


def __pad_real_im(img, padding_pix_pct=0.03, img_size=256, padding_value=255):
    padding_pix = int(img_size * padding_pix_pct)
    """
    make the bounding box larger by a few pixels (equiv. to
    padding_pix pixels after resize), then add edge padding
    to make it a square, then resize to desired resolution
    """
    y1, x1, y2, x2 = [0, 0, img.shape[1], img.shape[0]]
    w, h = img.shape[1], img.shape[0]
    x_mid = (x1 + x2) / 2.
    y_mid = (y1 + y2) / 2.
    ll = max(x2 - x1, y2 - y1) * img_size / (img_size - 2. * padding_pix)
    x1 = int(np.round(x_mid - ll / 2.))
    x2 = int(np.round(x_mid + ll / 2.))
    y1 = int(np.round(y_mid - ll / 2.))
    y2 = int(np.round(y_mid + ll / 2.))
    b_x = 0
    if x1 < 0:
        b_x = -x1
        x1 = 0
    b_y = 0
    if y1 < 0:
        b_y = -y1
        y1 = 0
    a_x = 0
    if x2 >= h:
        a_x = x2 - (h - 1)
        x2 = h - 1
    a_y = 0
    if y2 >= w:
        a_y = y2 - (w - 1)
        y2 = w - 1
    if len(img.shape) == 2:
        crop = np.pad(img[x1: x2 + 1, y1: y2 + 1],
                      ((b_x, a_x), (b_y, a_y)), mode='constant', constant_values=padding_value)
    else:
        crop = np.pad(img[x1: x2 + 1, y1: y2 + 1],
                      ((b_x, a_x), (b_y, a_y), (0, 0)), mode='constant', constant_values=padding_value)
    return crop


@numba.jit(nopython=True, cache=True)
def downsample(vox_in, times, use_max=True):
    if vox_in.shape[0] % times != 0:
        print('WARNING: not dividing the space evenly.')
    dim = vox_in.shape[0] // times
    vox_out = np.zeros((dim, dim, dim))
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                subx = x * times
                suby = y * times
                subz = z * times
                subvox = vox_in[subx:subx + times - 1,
                                suby:suby + times - 1, subz:subz + times - 1]
                if use_max:
                    vox_out[x, y, z] = np.max(subvox)
                else:
                    vox_out[x, y, z] = np.mean(subvox)
    return vox_out


def find_bound(voxel, *, threshold=0.5):
    """ find the boundary of a 3D voxel matrix. return boundaries in two matrices.
    Note that lower bound is inclusive while the higher bound is not"""
    assert voxel.ndim == 3
    bmin = np.zeros(voxel.ndim, dtype=int)
    bmax = np.zeros(voxel.ndim, dtype=int)

    voxel_binary = (voxel > threshold)
    if not voxel_binary.any():
        print(str_warning, 'Empty voxel found')
        return bmin, bmax

    for dim in range(voxel.ndim):
        voxel_dim = voxel_binary
        for i in range(dim):
            voxel_dim = voxel_dim.any(0)
        for i in range(voxel.ndim - dim - 1):
            voxel_dim = voxel_dim.any(1)
        inds = voxel_dim.nonzero()[0]
        bmin[dim] = inds.min()
        bmax[dim] = inds.max() + 1
    return bmin, bmax


def save_obj(vertices, faces, path):
    with open(path, 'w') as f:
        for v in vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in faces:
            f.write('f {} {} {}\n'.format(
                face[0] + 1, face[1] + 1, face[2] + 1))


def save_vox_to_obj(voxel, th, save_path):
    if len(voxel.shape) == 5:
        voxel_iso = voxel[0, 0, :, :, :]
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxel_iso, th, spacing=(1 / 128, 1 / 128, 1 / 128),)
    save_obj(verts - 0.5, faces, save_path)


def bounding_box_align(voxel, gt, *, threshold=0.5):
    bminv, bmaxv = find_bound(voxel, threshold=threshold)
    bming, bmaxg = find_bound(voxel, threshold=threshold)
    scale = (bmaxg - bming).max() / (bmaxv - bminv).max()
    scales = np.array((scale, scale, scale))
    offset = (bmaxg + bming) / 2 - (bmaxv + bminv) / 2
    return transform(voxel, scales=scales, offset=offset)


def translate(voxel, *, offset=None, translate_type=None):
    """ Translate a voxel by a specific offset """
    assert voxel.ndim == 3
    bmin, bmax = find_bound(voxel, threshold=0.5)
    if offset is None:
        assert translate_type is not None
        min_offset = -bmin
        max_offset = np.array(voxel.shape) - bmax
        if translate_type == 'random':
            offset = np.random.rand(
                voxel.ndim) * (max_offset - min_offset + 1) + min_offset - 1
            offset = np.ceil(offset).astype(int)
        elif translate_type == 'origin':
            offset = min_offset
        elif translate_type == 'middle':
            offset = np.ceil((min_offset + max_offset) / 2).astype(int)
        else:
            raise ValueError('unknown translate_type: ' + str(translate_type))
    else:
        assert translate_type is None
        offset = np.array(offset).astype(int)
    assert (bmin + offset >= 0).all() and (bmax +
                                           offset <= np.array(voxel.shape)).all()
    res = np.zeros(voxel.shape)
    res[bmin[0] + offset[0]:bmax[0] + offset[0], bmin[1] + offset[1]:bmax[1] + offset[1], bmin[2] +
        offset[2]:bmax[2] + offset[2]] = voxel[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]]
    return res


def dim_unify(voxel):
    if voxel.ndim == 5:
        assert voxel.shape[1] == 1
        voxel = voxel[:, 0, :, :, :]
    elif voxel.ndim == 3:
        voxel = np.expand_dims(voxel, 0)
    else:
        assert voxel.ndim == 4, 'voxel matrix must have dimensions of 3, 4, 5'
    return voxel

###########################################################
# For non-discretized transformation


def _get_centeralized_mesh_grid(sx, sy, sz):
    x = np.arange(sx) - sx / 2.
    y = np.arange(sy) - sy / 2.
    z = np.arange(sz) - sz / 2.
    return np.meshgrid(x, y, z, indexing='ij')


def get_rotation_matrix(angles):
    # # legacy code
    # alpha, beta, gamma = angles
    # R_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    # R_beta = np.array([[1, 0, 0], [0, np.cos(beta), -np.sin(beta)], [0, np.sin(beta), np.cos(beta)]])
    # R_gamma = np.array([[np.cos(gamma), 0, -np.sin(gamma)], [0, 1, 0], [np.sin(gamma), 0, np.cos(gamma)]])
    # R = np.dot(np.dot(R_alpha, R_beta), R_gamma)
    alpha, beta, gamma = angles
    R_alpha = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    R_beta = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    R_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    R = np.dot(np.dot(R_alpha, R_beta), R_gamma)
    return R


def get_scale_matrix(scales):
    return np.diag(scales)


def transform_by_matrix(voxel, matrix, offset):
    """
    transform a voxel by matrix, then apply an offset
    Note that the offset is applied after the transformation
    """
    sx, sy, sz = voxel.shape
    gridx, gridy, gridz = _get_centeralized_mesh_grid(sx, sy, sz)  # the coordinate grid of the new voxel
    mesh = np.array([gridx.reshape(-1), gridy.reshape(-1), gridz.reshape(-1)])
    mesh_rot = np.dot(np.linalg.inv(matrix), mesh) + np.array([sx / 2, sy / 2, sz / 2]).reshape(3, 1)
    mesh_rot = mesh_rot - np.array(offset).reshape(3, 1)    # grid for new_voxel should get a negative offset

    interp = rgi((np.arange(sx), np.arange(sy), np.arange(sz)), voxel,
                 method='linear', bounds_error=False, fill_value=0)
    new_voxel = interp(mesh_rot.T).reshape(sx, sy, sz)  # todo: move mesh to center
    return new_voxel


def transform(voxel, angles=(0, 0, 0), scales=(1, 1, 1), offset=(0, 0, 0), threshold=None, clamp=False):
    """
    transform a voxel by first rotate, then scale, then add offset.
    shortcut for transform_by_matrix
    """
    matrix = np.dot(get_rotation_matrix(angles), get_scale_matrix(scales))
    new_voxel = transform_by_matrix(voxel, matrix, offset)
    if clamp:
        new_voxel = np.clip(new_voxel, 0, 1)
    if threshold is not None:
        new_voxel = (new_voxel > threshold).astype(np.uint8)
    return new_voxel

############################################################
# test


def _fill(*, input_array, six_way=True):
    """
    fill an voxel array with dfs
    The algorithm pad the input_array with 2 voxels in each direction
        (1 to avoid complex border cases when checking neighbors,
         1 for making outer voxels connected so that we only search for one connected component)
    Note that this script considers ALL non-zero values as surface voxels
    """
    UNKNOWN = 200   # must be in [0, 255] for uint8

    sz0, sz1, sz2 = input_array.shape
    output_array = np.zeros((sz0 + 4, sz1 + 4, sz2 + 4), dtype=np.uint8)
    output_array[1:-1, 1:-1, 1:-1] = UNKNOWN
    input_padded = np.zeros((sz0 + 4, sz1 + 4, sz2 + 4), dtype=np.uint8)
    input_padded[2:-2, 2:-2, 2:-2] = input_array

    stack = [(1, 1, 1)]
    output_array[1, 1, 1] = 0
    while len(stack) > 0:
        i, j, k = stack.pop()
        output_array[i, j, k] = 0
        if six_way:
            neighbors = [(i - 1, j, k),
                         (i, j - 1, k),
                         (i, j, k - 1),
                         (i, j, k + 1),
                         (i, j + 1, k),
                         (i + 1, j, k), ]
        else:
            neighbors = [(i - 1, j - 1, k - 1),
                         (i - 1, j - 1, k),
                         (i - 1, j - 1, k + 1),
                         (i - 1, j, k - 1),
                         (i - 1, j, k),
                         (i - 1, j, k + 1),
                         (i - 1, j + 1, k - 1),
                         (i - 1, j + 1, k),
                         (i - 1, j + 1, k + 1),
                         (i, j - 1, k - 1),
                         (i, j - 1, k),
                         (i, j - 1, k + 1),
                         (i, j, k - 1),
                         (i, j, k + 1),
                         (i, j + 1, k - 1),
                         (i, j + 1, k),
                         (i, j + 1, k + 1),
                         (i + 1, j - 1, k - 1),
                         (i + 1, j - 1, k),
                         (i + 1, j - 1, k + 1),
                         (i + 1, j, k - 1),
                         (i + 1, j, k),
                         (i + 1, j, k + 1),
                         (i + 1, j + 1, k - 1),
                         (i + 1, j + 1, k),
                         (i + 1, j + 1, k + 1), ]
        for i_, j_, k_ in neighbors:
            if output_array[i_, j_, k_] == UNKNOWN and input_padded[i_, j_, k_] == 0:
                stack.append((i_, j_, k_))
                output_array[i_, j_, k_] = 0
    output_array = (output_array != 0).astype(np.uint8)
    return output_array[2:-2, 2:-2, 2:-2]


def fill(use_compile=False, compile_flag={'cache': True, 'nopython': True}, **kwargs):
    """
    common compile flags: {'cache': True, 'nopython': True}
    """
    if use_compile:
        from numba import jit
        return jit(**compile_flag)(_fill)(**kwargs)
    else:
        return _fill(**kwargs)


if __name__ == '__main__':
    import os
    from scipy.io import loadmat
    test_filename = '/path/to/model_normalized_128.mat'
    gt = loadmat(test_filename)['voxel']
    gt = downsample(gt, 4)
    print(gt.shape)
    results = [np.copy(gt)]     # 0

    # translation
    results.append(translate(gt, offset=(0, 0, 4)))   # 1
    results.append(translate(gt, offset=(0, 0, -4)))  # 2

    # transform_offset
    results.append(transform(gt, threshold=None))    # 3
    results.append(transform(gt, offset=(0, 0, 4), threshold=None))    # 4
    results.append(transform(gt, offset=(0, 0, -4), threshold=None))   # 5

    # transform_rotate
    results.append(transform(gt, angles=(1., 0, 0), threshold=None))  # 6
    results.append(transform(gt, angles=(0, 1., 0), threshold=None))  # 7
    results.append(transform(gt, angles=(0, 0, 1.), threshold=None))  # 8

    # transform_scale
    results.append(transform(gt, scales=(0.8, 0.8, 0.8), threshold=None))      # 9
    results.append(transform(gt, scales=(1.2, 1.2, 1.2), threshold=None, clamp=True))    # 10

    # check that gt is not changed
    assert (gt == results[0]).all()

    save_test_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'util_voxel_test.npz')
    np.savez_compressed(save_test_filename, voxels=np.stack(results))

    s = 'blender --background --python /path/to/voxel_render.py -- '
    s += ' --input %s ' % save_test_filename
    s += " --saveobj 1 --saveimg 0 --skimage 1 --threshold 0.5"  # note that this is visualization threshold, since all output are converted to 0/1
    print(s)
