import argparse


class Environment:
    def __init__(self):
        self.args = self._init_args()

    def _init_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--skip_n_examples', type=int, default=0,
                            help='Number of examples to skip from the dataset.')
        args, _ = parser.parse_known_args()
        return args
