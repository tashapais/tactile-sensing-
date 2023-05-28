import torch

# Create a tensor with dimensions (x, y, z)
tensor_xyz = torch.randn(2, 3, 4)

# Transpose the tensor to dimensions (z, x, y)
tensor_zxy = tensor_xyz.permute(2, 0, 1)

# Print the original and transposed tensors
print("Original tensor (x, y, z):\n", tensor_xyz.shape)
print("Transposed tensor (z, x, y):\n", tensor_zxy.shape)



import torch

# Assume 'img' is your tensor of shape (3, 32, 32)
img = torch.randn(3, 32, 32)

# To get the RGB value of the pixel at location (x, y)
x = 15
y = 20

rgb_value = img[:, x, y]

print(rgb_value)