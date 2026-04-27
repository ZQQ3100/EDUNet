from opts.base_opt import BaseOpt

class TestOpt(BaseOpt):
    def initialize(self):
        BaseOpt.initialize(self)
        self.parser.add_argument('--input_event_path', type=str, default='../test_data/gopro_test/eventstream_mat/')
        self.parser.add_argument('--input_blur_path', type=str, default='../test_data/gopro_test/blur_images/')
        # augmentation.
        self.parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--output_dir', type=str, default='../result/gopro_results', help='output_path')
        self.parser.add_argument('--load_G', type=str, default='../Checkpoint/v8_540_10_KKT/model_790.pth', help='path of the pretrained model')
        # use cuda
        self.parser.add_argument("--cuda", type=str, default='cuda:0')

        self.is_train = False
