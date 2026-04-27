import argparse

class BaseOpt():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # dataset
        self.parser.add_argument('--dataset_mode', type=str, default='gopro', help='gopro, revd')
        # dataloader
        self.parser.add_argument('--n_threads', default=4, type=int, help='# threads for data')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.channel = 1

        self.opt.is_train = self.is_train

        return self.opt
