import torch

if __name__ == '__main__':
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1904)
    print(torch.randint(0, 1, (2,),generator=None))