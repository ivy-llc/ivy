Training a Network
==================

So far, we have only considered using Ivy to create framework agnostic functions and libraries,
which can then be used alongside framework-specific code, such as a PyTorch project.

Ivy can also be used to create a fully framework agnostic training pipeline, including stateful neural network models and optimizers.

Trainable Ivy Module
--------------------

A trainable Ivy module can be constructed like so:

.. code-block:: python

    class IvyFcModel(ivy.Module):

        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear2 = ivy.Linear(64, 1)
            ivy.Module.__init__(self, 'cpu')

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear2(x))

This model is now ready to train!

Ivy Training
------------

This model can then be trained using Ivy like so:

.. code-block:: python

    ivy.set_framework('torch')  # change to any framework
    model = IvyFcModel()
    optimizer = ivy.Adam(1e-3)
    x_in = ivy.array([1., 2., 3.])


    def loss_fn(v):
        return ivy.reduce_mean(model(x_in, v=v))[0]


    for step in range(100):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
        print('step {} loss = {}'.format(step, ivy.to_numpy(loss).item()))

This example does not work with a NumPy backend, as NumPy does not support gradients.
With any other framework, you can watch the loss quickly go down!

Native Training
---------------

Alternatively, the same Ivy network model can be trained with optimizer classes from your favourite framework.

**PyTorch**

This model can be trained using PyTorch like so:

.. code-block:: python

    class TorchFcModel(torch.nn.Module, IvyFcModel):

        def __init__(self):
            torch.nn.Module.__init__(self)
            IvyFcModel.__init__(self)
            self._assign_variables()

        def _assign_variables(self):
            self.v.map(
                lambda x, kc: self.register_parameter(name=kc, param=torch.nn.Parameter(x)))
            self.v = self.v.map(lambda x, kc: self._parameters[kc])

        def forward(self, x):
            return self._forward(x)


    ivy.set_framework('torch')
    model = TorchFcModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_in = torch.tensor([1., 2., 3.])


    def loss_fn():
        return torch.mean(model(x_in))


    for step in range(100):
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        print('step {} loss = {}'.format(step, ivy.to_numpy(loss).item()))
