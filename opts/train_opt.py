from opts.base_opt import BaseOpt


class TrainOpt(BaseOpt):
    def initialize(self):
        BaseOpt.initialize(self)
        self.parser.add_argument('--input_event_path', type=str, default='../train_data/gopro_train/eventstream_mat/',
                                 help='Path to input event data')
        self.parser.add_argument('--input_blur_path', type=str, default='../train_data/gopro_train/blur_images/',
                                 help='Path to input blurry images')
        self.parser.add_argument('--input_gt_path', type=str, default='../train_data/gopro_train/gt/',
                                 help='Path to input ground truth images')

        # Training options
        self.parser.add_argument('--train_batch_size', type=int, default=2, help='Batch size for training')
        self.parser.add_argument('--num_epochs', type=int, default=800, help='Number of epochs for training')
        self.parser.add_argument('--num_workers', type=int, default=2)
        self.parser.add_argument('--learning_rate', type=float, default=0.0003, help='Initial learning rate')


        # Use CUDA

        self.parser.add_argument("--cuda", type=str, default='cuda:0')

        self.is_train = True