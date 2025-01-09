from control_flow_experimental.autograph_ivy.core.api import to_functional_form
import ivy

IVY_BACKENDS = ["numpy", "torch", "jax", "tensorflow", "paddle"]


def test_function(func_class, data, *constructor_args, backends=IVY_BACKENDS):
    for backend in backends:
        # Set the backend
        ivy.set_backend(backend)

        # fix the seed value to get deterministic outputs
        ivy.seed(seed_value=42)

        # Create an instance of the model class
        model = func_class(*constructor_args)

        # Create model inputs
        inputs = ivy.array(data, dtype=ivy.float32)

        # Run the original function and get the output
        output = model(inputs)
        print(f"Original function output ({backend} backend): {output}")

        # Convert the function to its functional form
        converted_func = to_functional_form(model._forward)

        # Run the converted function and get its output
        converted_output = converted_func(inputs)
        print(f"Converted function output ({backend} backend): {converted_output}")

        # Perform the assertion that the outputs are the same
        assert ivy.array_equal(
            output, converted_output
        ), f"Outputs do not match for function {model.__name__} and backend {backend}"


class SimpleNeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.weights_input_to_hidden = ivy.random.random_normal(
            mean=0, std=1, shape=(num_inputs, num_hidden)
        )
        self.biases_input_to_hidden = ivy.zeros(num_hidden)
        self.weights_hidden_to_output = ivy.random.random_normal(
            mean=0, std=1, shape=(num_hidden, num_outputs)
        )
        self.biases_hidden_to_output = ivy.zeros(num_outputs)

    def forward(self, inputs):
        hidden_inputs = (
            ivy.matmul(inputs, self.weights_input_to_hidden)
            + self.biases_input_to_hidden
        )
        hidden_outputs = ivy.zeros(hidden_inputs.shape)
        i = 0
        while i < hidden_inputs.shape[0]:
            if any(hidden_inputs[i]) > 0:
                hidden_outputs[i] = 1
            else:
                hidden_outputs[i] = 0
            i += 1
        output = (
            ivy.matmul(hidden_outputs, self.weights_hidden_to_output)
            + self.biases_hidden_to_output
        )
        return output


class DynamicCNN(ivy.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.conv1 = ivy.Conv2D(
            num_channels, 32, [3, 3], 1, "VALID", data_format="NCHW"
        )
        self.conv2 = ivy.Conv2D(32, 64, [3, 3], 1, "VALID", data_format="NCHW")
        self.conv3 = ivy.Conv2D(64, 128, [3, 3], 1, "VALID", data_format="NCHW")
        self.fc1 = ivy.Linear(128 * 10 * 10, 512)
        self.fc2 = ivy.Linear(512, num_classes)

    def _forward(self, x, use_relu=True):
        i = 0
        while i < 3:
            if i == 0:
                x = self.conv1(x)
            elif i == 1:
                x = self.conv2(x)
            else:
                x = self.conv3(x)
            if use_relu:
                x = ivy.relu(x)
            i += 1

        x = ivy.reshape(x, (-1, 128 * 10 * 10))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Test the function (uncomment any one)

# inputs = ivy.random.random_normal(mean=0, std=1, shape=(1,3,16,16))
# test_function(DynamicCNN, inputs, 3, 10)


# inputs = ivy.array([[1, 2, 3]], dtype=ivy.float32)
# test_function(SimpleNeuralNetwork, inputs, 3,5,2)
