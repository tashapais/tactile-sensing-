import torch





if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
