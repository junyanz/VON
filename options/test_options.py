from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='../results/', help='saves results here.')
        parser.add_argument('--n_shapes', type=int, default=10, help='the number of sampled shapes')
        parser.add_argument('--n_views', type=int, default=10, help='the number of sampled views')
        parser.add_argument('--reset_shape', action='store_true', help='sample a different shape')
        parser.add_argument('--reset_texture', action='store_true', help='sample a different texture')
        parser.add_argument('--real_shape', action='store_true', help='use real voxels')
        parser.add_argument('--real_texture', action='store_true', help='use real textures')
        parser.add_argument('--render_3d', action='store_true', help='use blender to render 3d')
        parser.add_argument('--render_25d', action='store_true', help='use blender to render 2.5d')
        parser.add_argument('--random_view', action='store_true', help='show random views')
        parser.add_argument('--show_input', action='store_true', help='show input image')
        parser.add_argument('--interp_shape', action='store_true', help='interpolate in shape space')
        parser.add_argument('--interp_texture', action='store_true', help='interpolate in texture space')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for the results')
        parser.add_argument('--ios_th', default=0.01, type=float, help='thresholding for isosurface')

        self.isTrain = False
        return parser
