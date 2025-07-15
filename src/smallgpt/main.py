import torch
from model import SimpleNN


def main():
    # Define dummy input data
    input_dim = 10
    output_dim = 2
    batch_size = 5

    x = torch.randn(batch_size, input_dim)
    print("Input:")
    print(x)

    # Initialize model
    model = SimpleNN(input_dim, output_dim)

    # Run forward pass
    output = model(x)
    print("Output:")
    print(output)


if __name__ == "__main__":
    main()