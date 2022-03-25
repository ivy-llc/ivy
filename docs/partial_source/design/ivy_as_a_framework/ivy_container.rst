Ivy Container
=============

In this post, we’ll be explaining how the *ivy.Container* class save you a ton of time and clean up code in almost all aspects of your ML workflow. So without further ado, let’s dive in!

Firstly, Dictionaries are an incredibly powerful and useful data type in Python. They enable a clean, readable and efficient-access (via hashing) storage of arbitrarily hierarchical data.

The *ivy.Container* class can be seen as a souped-up Dict, with many useful features built on top. It’s the backbone of most high level operations in Ivy.

Let’s walk through some of the most important features of the *ivy.Container*!

Construction
------------

A container can be constructed in a number of ways. All construction approaches below result in identical *ivy.Container* instances.

.. code-block:: python

   import ivy
   ivy.set_framework('torch')

    dct = {'a': ivy.array([0.]),
           'b': {'c': ivy.array([1.]),
                 'd': ivy.array([2.])}}

   # via dict
   cnt = ivy.Container(dct)

   # via keyword
   cnt = ivy.Container(a=ivy.array([0.]),
                       b=ivy.Container(c=ivy.array([1.]),
                                       d=ivy.array([2.])))

   # combos
   cnt = ivy.Container(a=ivy.array([0.]),
                       b={'c': ivy.array([1.]),
                          'd': ivy.array([2.])})

   cnt = ivy.Container({'a': ivy.array([0.]),
                        'b': ivy.Container(c=ivy.array([1.]),
                                           d=ivy.array([2.]))})

Representation
--------------

*ivy.Container* prints the hierarchical structure to the terminal in a very intuitive manner, much more so than native Python Dicts.

.. code-block:: python

   print(dct)

   {'a': tensor([0.], device='cuda:0'), 'b'
   : {'c': tensor([1.], device='cuda:0'), '
   d': tensor([2.], device='cuda:0')}}

   print(cnt)

   {
       a: tensor([0.], device=cuda:0),
       b: {
           c: tensor([1.], device=cuda:0),
           d: tensor([2.], device=cuda:0)
       }
   }

If the container holds very large arrays, then their shapes are printed instead. Again, this does not happen with native Python Dicts.

.. code-block:: python

   dct = {'a': ivy.ones((1000, 3)),
          'b': {'c': ivy.zeros((3, 1000)),
                'd': ivy.ones((1000, 2))}}

   print(dct)

   {'a': tensor([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.],
           ...,
           [1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], device='cuda:0'), 'b': {'c'
   : tensor([[0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.]], device='c
   uda:0'), 'd': tensor([[1., 1.],
           [1., 1.],
           [1., 1.],
           ...,
           [1., 1.],
           [1., 1.],
           [1., 1.]], device='cuda:0')}}

   cnt = ivy.Container(dct)

   print(cnt)

   {
       a: (<class torch.Tensor> shape=[1000, 3]),
       b: {
           c: (<class torch.Tensor> shape=[3, 1000]),
           d: (<class torch.Tensor> shape=[1000, 2])
       }
   }

Recursive Methods
----------------

All methods in Ivy’s functional API are implemented as recursive methods on the *ivy.Container*. This means you can easily map a single method to all arrays in the container with a single line.

Starting with the following container:

.. code-block:: python

   cnt = ivy.Container({'a': ivy.array([0., 1., 2.]),
                        'b': {'c': ivy.array([2., 6., 5.]),
                              'd': ivy.array([10., 5., 2.])}})

We can compute the mean of each sub-array:

.. code-block:: python

   print(cnt.reduce_mean())

   {
       a: tensor([1.], device=cuda:0),
       b: {
           c: tensor([4.3333], device=cuda:0),
           d: tensor([5.6667], device=cuda:0)
       }
   }

Or we can flip each sub-array:

.. code-block:: python

   print(cnt.flip())

   {
       a: tensor([2., 1., 0.], device=cuda:0),
       b: {
           c: tensor([5., 6., 2.], device=cuda:0),
           d: tensor([2., 5., 10.], device=cuda:0)
       }
   }

There are 178 such functions for the *ivy.Container* class in total, check out the `code <https://github.com/unifyai/ivy/blob/master/ivy/container.py>`_ or `docs <https://lets-unify.ai/ivy/core/container.html>`_ to see what they are!

Built-ins
----------

All built-in methods also apply recursively. For example, performing a gradient update step for a set of network weights can be done in one line.

.. code-block:: python

   weights = ivy.Container(
          {'linear': {'b': ivy.array([0.2]),
                      'w': ivy.array([1.5, 2.3, 0.9])}})

   grads = ivy.Container(
          {'linear': {'b': ivy.array([1.4]),
                      'w': ivy.array([1.9, 0.6, 2.1])}})

   lr = 0.1

   new_weights = weights - grads * lr
   print(new_weights)

   {
       linear: {
           b: tensor([0.0600], device=cuda:0),
           w: tensor([1.3100, 2.2400, 0.6900], device=cuda:0)
       }
   }

Check out the section below on Ivy’s stateful API to see how the *ivy.Container* is used for storing all network weights in *ivy.Module* instances!

Access
------

The keys in an *ivy.Container* can be set and accessed by using either class attributes or keys in the dictionary. Both of these setting and accessing approaches are equivalent under the hood.

.. code-block:: python

   cnt = ivy.Container({'a': ivy.array([0.])})

   cnt['b'] = ivy.array([1.])
   cnt.c = ivy.array([2.])

   print(cnt)

   {
       a: tensor([0.], device=cuda:0),
       b: tensor([1.], device=cuda:0),
       c: tensor([2.], device=cuda:0)
   }

   assert cnt.c is cnt['c']

Nested keys can also be set in one line, using either ‘/’ or ‘.’ as a delimiter.

.. code-block:: python

   cnt = ivy.Container({'a': ivy.array([0.])})
   cnt['b/c'] = ivy.array([1.])
   cnt['d.e.f'] = ivy.array([2.])

   print(cnt)

   {
       a: tensor([0.], device=cuda:0),
       b: {
           c: tensor([1.], device=cuda:0)
       },
       d: {
           e: {
               f: tensor([2.], device=cuda:0)
           }
       }
   }

One of the key benefits of using properties under the hood is the autocomplete support this introduces. Class attributes can be auto-completed when pressing tab midway through typing. This is not possible with Dicts.

.. code-block:: python

   cnt = ivy.Container({'agent': {'total_speed': ivy.array([0.])}})
   cnt.agent.total_height = ivy.array([1.])
   cnt['agent/total_width'] = ivy.array([2.])

   cnt.age -> tab
   cnt.agent
   cnt.agent.tot -> tab
   cnt.agent.total_ -> tab

   cnt.agent.total_height  cnt.agent.total_speed   cnt.agent.total_width

   cnt.agent.total_h -> tab
   cnt.agent.total_height

   tensor([1.], device='cuda:0')

Saving and Loading
------------------

Saving and loading to disk can be done in one of many ways, with each being suited to different data types in the container.

For example, if the container mainly contains arrays (such as the weights of a network), then one of the following can be used.

.. code-block:: python

   weights = ivy.Container(
          {'linear': {'b': ivy.array([[0.2]]),
                      'w': ivy.array([[1.5, 2.3, 0.9]])}})

   # save and load as hdf5
   weights.to_disk_as_hdf5('weights.hdf5')
   loaded = ivy.Container.from_disk_as_hdf5('weights.hdf5')
   assert ivy.Container.identical(
          [loaded, weights], same_arrays=False)

   # save and load as pickled
   weights.to_disk_as_pickled('weights.pickled')
   loaded = ivy.Container.from_disk_as_pickled('weights.pickled')
   assert ivy.Container.identical(
          [loaded, weights], same_arrays=False)

Alternatively, if the container mainly stored experiment configuration data, then the following can be used.

.. code-block:: python

   config = ivy.Container(
          {'loading': {'batch_size': 16,
                       'dir': '/dataset/images'},
           'training': {'dropout': True,
                        'lr': 0.1,
                        'optim': 'ADAM'}})

   # save and load as json
   config.to_disk_as_json('config.json')

   # config.json contents -------------#
   # {                                 #
   #     "loading": {                  #
   #         "batch_size": 16,         #
   #         "dir": "/dataset/images"  #
   #     },                            #
   #     "training": {                 #
   #         "dropout": true,          #
   #         "lr": 0.1,                #
   #         "optim": "ADAM"           #
   #     }                             #
   # }                                 #
   # ----------------------------------#

   loaded = ivy.Container.from_disk_as_json('config.json')
   assert (config == loaded).all_true()

Comparisons
-----------

Comparing differences between containers can be achieved on a per-leaf basis. This is useful for debugging and also comparing configurations between runs. For example, consider a case where two containers of arrays should be identical at all levels. We can then very quickly find conflicting leaves.

.. code-block:: python

   cnt0 = ivy.Container({'a': ivy.array([0.]),
                      'b': ivy.array([1.])})
   cnt1 = cnt0.deep_copy()
   cnt1.b = ivy.array([0.])

   print(ivy.Container.diff(cnt0, cnt1))

   {
       a: tensor([0.], device=cuda:0),
       b: {
           diff_0: tensor([1.], device=cuda:0),
           diff_1: tensor([0.], device=cuda:0)
       }
   }

Or perhaps we saved JSON configuration files to disk for two different experiment runs, and then want to quickly see their differences. The *ivy.Container.diff* method will also detect differences in the hierarchical structure and key name differences.

.. code-block:: python

    config0 = ivy.Container(
           {'batch_size': 8,
            'lr': 0.1,
            'optim': 'ADAM'})

    config1 = ivy.Container(
           {'batch_size': 16,
            'dropout': 0.5,
            'lr': 0.1})

    print(ivy.Container.diff(config0, config1))

    {
        batch_size: {
            diff_0: 8,
            diff_1: 16
        },
        dropout: {
            diff_1: 0.5
        },
        lr: 0.1,
        optim: {
            diff_0: ADAM
        }
    }

The *ivy.Container.diff* method can be applied to arbitrarily many containers at once in a single call, not just two as in the examples above.

Customized Representations
-------------------------

Not only does *ivy.Container* print to the terminal in a very intuitive manner, but there are also helper functions to fully control this representation. This is very helpful when debugging networks with huge numbers of parameters with a deep hierarchical structure for example.

If our networks weights go many levels deep in the nested hierarchy, we might not want to see all of them when printing our container to screen. Consider the following nested structure.

.. code-block:: python

    weights = ivy.Container(
        {'decoder':
             {'l0':
                  {'b': ivy.array([0.]),
                   'w': ivy.array([[0.]])},
              'l1':
                  {'b': ivy.array([0.]),
                   'w': ivy.array([[0.]])}},
         'encoder':
             {'l0':
                  {'b': ivy.array([0.]),
                   'w': ivy.array([[0.]])},
              'l1':
                  {'b': ivy.array([0.]),
                   'w': ivy.array([[0.]])}},
         'l0':
             {'b': ivy.array([0.]),
              'w': ivy.array([[0.]])},
         'l1':
             {'b': ivy.array([0.]),
              'w': ivy.array([[0.]])}})

We can clip the depth of the printed container in order to make the structure of the root keys clearer. All nested structures below this depth are truncated into single keys with a “__” delimiter appending all keys below this depth.

.. code-block:: python

    weights.flatten_key_chains(above_height=1)

    {
        decoder__l0: {
            b: tensor([0.], device=cuda:0),
            w: tensor([[0.]], device=cuda:0)
        },
        decoder__l1: {
            b: tensor([0.], device=cuda:0),
            w: tensor([[0.]], device=cuda:0)
        },
        encoder__l0: {
            b: tensor([0.], device=cuda:0),
            w: tensor([[0.]], device=cuda:0)
        },
        encoder__l1: {
            b: tensor([0.], device=cuda:0),
            w: tensor([[0.]], device=cuda:0)
        },
        l0: {
            b: tensor([0.], device=cuda:0),
            w: tensor([[0.]], device=cuda:0)
        },
        l1: {
            b: tensor([0.], device=cuda:0),
            w: tensor([[0.]], device=cuda:0)
        }
    }


Likewise, we can clip the height of the printed container in order to make the structure of the leaf keys clearer. All nested structures above this height are truncated into single keys with a “__” delimiter appending all keys above this height.

.. code-block:: python

    weights.flatten_key_chains(below_depth=1)

    {
        decoder: {
            l0__b: tensor([0.], device=cuda:0),
            l0__w: tensor([[0.]], device=cuda:0),
            l1__b: tensor([0.], device=cuda:0),
            l1__w: tensor([[0.]], device=cuda:0)
        },
        encoder: {
            l0__b: tensor([0.], device=cuda:0),
            l0__w: tensor([[0.]], device=cuda:0),
            l1__b: tensor([0.], device=cuda:0),
            l1__w: tensor([[0.]], device=cuda:0)
        },
        l0: {
            b: tensor([0.], device=cuda:0),
            w: tensor([[0.]], device=cuda:0)
        },
        l1: {
            b: tensor([0.], device=cuda:0),
            w: tensor([[0.]], device=cuda:0)
        }
    }

These are very useful methods when stepping through code and debugging complex nested structures such as the weights of a network.

There are also methods: *with_print_limit* for controlling the printable size of arrays before the shape is instead displayed, *with_key_length_limit* for setting the maximum key length before string clipping, *with_print_indent* for controlling the nested indent, and many more. Check out the `docs <https://lets-unify.ai/ivy/core/container.html>`_ for more details!

Use Cases
---------

We’ll now just go through a few of the different use cases for the Ivy Container. The container is not limited to these use cases though, the container is the right choice whenever you are storing nested data!

Compartmentalization
--------------------

The most obvious use case for the *ivy.Container* class is to compartmentalize inputs into a useful structure. For example, without better foresight, we could untidily implement a function *update_agent* as follows:

.. code-block:: python

    def normalize_img(img):
        img_max = ivy.reduce_max(img)
        img_min = ivy.reduce_min(img)
        img_range = img_max - img_min
        return (img - img_min) / img_range

    def update_agent(agent_position, agent_velocity,
                     agent_cam_front_rgb, agent_cam_front_depth,
                     agent_cam_rear_rgb, agent_cam_rear_depth,
                     agent_cam_lidar):

        # update agent state
        agent_position += ivy.array([0., 1., 2.])
        agent_velocity -= ivy.array([2., 1., 0.])

        # normalize images
        agent_cam_front_rgb = normalize_img(agent_cam_front_rgb)
        agent_cam_front_depth = normalize_img(agent_cam_front_depth)
        agent_cam_rear_rgb = normalize_img(agent_cam_rear_rgb)
        agent_cam_rear_depth = normalize_img(agent_cam_rear_depth)
        agent_cam_lidar = normalize_img(agent_cam_lidar)

        # return
        return agent_position, agent_velocity, agent_cam_front_rgb,\
               agent_cam_front_depth, agent_cam_rear_rgb,\
               agent_cam_rear_depth, agent_cam_lidar

Our code will be much cleaner if we do something like the following, particularly if there are many additional similar functions performing operations on the agent and the images:

.. code-block:: python

    class Cameras(ivy.Container):

        def __init__(self, front_rgb: ivy.Array, front_depth: ivy.Array,
                     rear_rgb: ivy.Array, rear_depth: ivy.Array,
                     lidar: ivy.Array):
            super().__init__(self,
                             front={'rgb': front_rgb,
                                    'depth': front_depth},
                             rear={'rgb': rear_rgb,
                                   'depth': rear_depth},
                             lidar=lidar)

    class Agent(ivy.Container):

        def __init__(self, position: ivy.Array,
                     velocity: ivy.Array, cams: Cameras):
            super().__init__(self, position=position,
                             velocity=velocity, cams=cams)

    def update_agent(agent: Agent):

        # update agent state
        agent.position += ivy.array([0., 1., 2.])
        agent.velocity -= ivy.array([2., 1., 0.])

        # normalize images
        cam_max = agent.cams.reduce_max()
        cam_min = agent.cams.reduce_min()
        cam_range = cam_max - cam_min
        agent.cams = (agent.cams - cam_min) / cam_range

Of course, this argument holds for the use of custom classes or built-in containers (Python list, dict, tuple etc.), and isn’t only relevant for the Ivy container. However the recursive methods of the Ivy Container make things even more convenient, such as where we recursively normalize all five images in the final four lines of the *update_agent* method.

Configuration
--------------

As briefly alluded to when explaining the *ivy.Container.diff* method, the container class is also the ideal data type for storing experiment configurations. Configurations can either first be stored to disk as a JSON file and then loaded into the *ivy.Container* for recursive comparisons to see differences between experiments, or the config can be specified in the code and then saved to disk as a JSON to keep a permanent log afterwards.

Data loading
-----------

The container can also be used for data loading. Our example uses single threaded loading, but incorporating multiprocessing with Queues is also pretty straightforward.

To start with, let’s assume we have an image Dataset saved to disk with separate images for a front camera and a rear camera for each point in time.

We can then load this Dataset with a configurable batch size like so, and we can easily iterate between each item in the batch. This is useful if we need to recursively unroll the entire batch in the time dimension for example.

.. code-block:: python

    class DataLoader:

        def __init__(self, batch_size):
            self._cnt = ivy.Container(
                dict(imgs={'front': 'images/front/img_{}.png',
                           'rear': 'images/rear/img_{}.png'}))
            self._dataset_size = 8
            self._batch_size = batch_size
            self._count = 0

        def __next__(self):
            cnt = self._cnt.copy()

            # image filenames
            img_fnames = ivy.Container.list_stack(
                [cnt.imgs.map(
                    lambda fname, _: fname.format(self._count + i)
                ) for i in range(self._batch_size)], 0
            )

            # load from disk
            loaded_imgs = img_fnames.map(
                lambda fnames, _: np.concatenate(
                    [np.expand_dims(cv2.imread(fname, -1), 0)
                     for fname in fnames], 0
                )
            ).from_numpy()

            # update count
            self._count += self._batch_size
            self._count %= self._dataset_size

            # return batch
            cnt.imgs = loaded_imgs
            return cnt

    loader = DataLoader(2)

    for _ in range(100):
        batch = next(loader)
        assert batch.imgs.front.shape == (2, 32, 32, 3)
        assert batch.imgs.rear.shape == (2, 32, 32, 3)
        for batch_slice in batch.unstack(0):
            assert batch_slice.imgs.front.shape == (32, 32, 3)
            assert batch_slice.imgs.rear.shape == (32, 32, 3)

Network weights
--------------

Finally, the Ivy Containers can also be used for storing network weights. In fact, as will be discussed in the next post on the stateful Ivy API, this is how the *ivy.Module* class stores all trainable variables in the model. The following code is possible thanks to the recursive operation of the container, which applies the gradient update to all variable arrays in the container recursively.

.. code-block:: python

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))

    model = MyModel()
    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])
    lr = 0.001

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.reduce_mean((out - target)**2)[0]

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(
          loss_fn, model.v)
        model.v = model.v - lr * grads
        print('step {} loss {}'.format(
          step, ivy.to_numpy(loss).item()))

    print(model.v)

    {
        linear0: {
            b: (<class torch.Tensor> shape=[64]),
            w: (<class torch.Tensor> shape=[64, 3])
        },
        linear1: {
            b: tensor([-0.0145], grad_fn=<AddBackward0>),
            w: (<class torch.Tensor> shape=[1, 64])
        }
    }

**Round Up**

That should hopefully be enough to get you started with the *ivy.Container* class 😊

Please check out the discussions on the `repo <https://github.com/unifyai/ivy>`_ for FAQs, and reach out on `discord <https://discord.gg/ZVQdvbzNQJ>`_ if you have any questions!