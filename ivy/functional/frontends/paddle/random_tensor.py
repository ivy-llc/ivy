import paddle
import paddle.nn.functional as F

# Generate a random tensor
tsr = paddle.randn([3, 4, 4])  # Shape: [channels, height, width]

# Normalize the tensor
normalized_tensor = F.normalize(tensor, mean=paddle.to_tensor([0.5, 0.5, 0.5]), std=paddle.to_tensor([0.5, 0.5, 0.5]))

# Perform a 2D max pooling operation
pooled_tensor = F.max_pool2d(normalized_tensor, kernel_size=2, stride=2)

# Apply 2D average pooling
averaged_tensor = F.avg_pool2d(normalized_tensor, kernel_size=2, stride=2)

# Apply a 2D convolution operation
weights = paddle.randn([3, 3, 3, 3])  # Shape: [output_channels, input_channels, kernel_size, kernel_size]
convolved_tensor = F.conv2d(normalized_tensor.unsqueeze(0), weights, stride=1, padding=1)

# Apply a batch normalization
normalized_batch = F.batch_norm(tensor.unsqueeze(0), running_mean=paddle.zeros([3]), running_var=paddle.ones([3]))

# Perform a 2D transposed convolution
weights_transposed = paddle.randn([3, 3, 3, 3])  # Shape: [output_channels, input_channels, kernel_size, kernel_size]
transposed_convolved_tensor = F.conv_transpose2d(normalized_tensor.unsqueeze(0), weights_transposed, stride=2, padding=1)

# Apply 2D dropout
dropout_probability = 0.2
dropout_tensor = F.dropout2d(tensor.unsqueeze(0), p=dropout_probability)

# Display the results
print("Original Tensor:")
print(tensor)
print("Normalized Tensor:")
print(normalized_tensor)
print("Max Pooled Tensor:")
print(pooled_tensor)
print("Averaged Tensor:")
print(averaged_tensor)
print("Convolved Tensor:")
print(convolved_tensor.squeeze(0))
print("Normalized Batch:")
print(normalized_batch.squeeze(0))
print("Transposed Convolved Tensor:")
print(transposed_convolved_tensor.squeeze(0))
print("Dropout Tensor:")
print(dropout_tensor.squeeze(0))

