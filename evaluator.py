import torch


class Evaluator:
    def __init__(self, explorer_path, discriminator_path):
        self.discriminator = None
        self.explorer = None
        self.explorer_path = explorer_path
        self.discriminator_path = discriminator_path

    def load_models(self):
        self.explorer = torch.load(self.explorer_path)
        self.discriminator = torch.load(self.discriminator_path)

    def evaluation(self):
        pass


if __name__ == "__main__":
    e = Evaluator("SAVED_MODELS/EXPLORER", "SAVED_MODELS/DISCRIMINATOR")
