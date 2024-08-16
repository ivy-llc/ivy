<div style="display: block;" align="center">
    <img class="only-dark" width="50%" src="https://raw.githubusercontent.com/ivy-llc/ivy-llc.github.io/main/src/assets/full_logo_dark_long.svg#gh-dark-mode-only"/>
</div>

<div style="display: block;" align="center">
    <img class="only-light" width="50%" src="https://raw.githubusercontent.com/ivy-llc/ivy-llc.github.io/main/src/assets/full_logo_light_long.svg#gh-light-mode-only"/>
</div>

------------------------------------------------------------------------

<div style="display: block;" align="center">
    <a href="https://ivy.dev/">
        Website
    </a>
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <a href="https://ivy.dev/docs">
        Docs
    </a>
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <a href="https://ivy.dev/docs/demos">
        Examples
    </a>
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <a href="https://ivy.dev/docs/overview/design.html">
        Design
    </a>
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <a href="https://ivy.dev/docs/overview/faq.html">
        FAQ
    </a>
</div>

<br>

<div style="margin-top: 10px; margin-bottom: 10px; display: block;" align="center">
    <a href="https://github.com/ivy-llc/ivy/issues">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/issues/ivy-llc/ivy">
    </a>
    <a href="https://github.com/ivy-llc/ivy/network/members">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/forks/ivy-llc/ivy">
    </a>
    <a href="https://github.com/ivy-llc/ivy/stargazers">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/ivy-llc/ivy">
    </a>
    <a href="https://github.com/ivy-llc/ivy/pulls">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
    </a>
    <a href="https://pypi.org/project/ivy">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy.svg">
    </a>
    <a href="https://github.com/ivy-llc/ivy/actions?query=workflow%3Adocs">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://github.com/ivy-llc/ivy/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://discord.gg/r5mcSAfp">
        <img class="dark-light" style="padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/1220325004013604945?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
</div>
<br clear="all" />


# Convert Machine Learning Code Between Frameworks

Ivy is an open-source machine learning framework that enables you to:

- Convert ML models, tools and libraries between frameworks while maintaining complete functionality using `ivy.transpile`
- Create optimized graph-based models and functions in any native framework (PyTorch, TensorFlow, etc..) with `ivy.trace_graph`
- Use your ML models or functions in any framework using a graph-tracing approach with `ivy.graph_transpile` *(deprecated)*

<div style="display: block;" align="center">
    <div>
    <a href="https://jax.readthedocs.io">
        <img class="dark-light" width="10%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/jax_logo.png">
    </a>
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <a href="https://www.tensorflow.org">
        <img class="dark-light" width="10%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/tensorflow_logo.png">
    </a>
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <a href="https://pytorch.org">
        <img class="dark-light" width="10%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/pytorch_logo.png">
    </a>
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <img class="dark-light" width="5%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    <a href="https://numpy.org">
        <img class="dark-light" width="10%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/numpy_logo.png">
    </a>
    </div>
</div>

<br clear="all" />

# Installing ivy

The easiest way to set up Ivy is to install it using **pip**:

``` bash
pip install ivy
```

<details>
<summary><b>Docker Image</b></summary>

You can pull the Docker image for Ivy from:

``` bash
docker pull ivyllc/ivy:latest
```

</details>

<details>
<summary><b>From Source</b></summary>

You can also install Ivy from source if you want to take advantage of
the latest changes, but we can\'t ensure everything will work as
expected 😅

``` bash
git clone https://github.com/ivy-llc/ivy.git
cd ivy
pip install --user -e .
```

If you want to set up testing and various frameworks it\'s probably     best
to check out the [Setting Up](https://ivy.dev/docs/overview/contributing/setting_up.html)
page, where OS-specific and IDE-specific instructions and video
tutorials to do so are available!

</details>

<br>

# Getting started

- [Docs](https://ivy.dev/docs)
- [Demos](https://ivy.dev/demos)
- [FAQ](https://ivy.dev/docs/overview/faq.html)

[Ivy's transpiler](https://ivy.dev/docs/overview/design/ivy_as_a_transpiler.html) allows you convert code between different ML frameworks. Have a look at our [Quickstart](https://ivy.dev/docs/demos/quickstart.html) notebook to get a brief idea of the features!

Beyond that, based on the frameworks you want to convert code between, there are a few more [examples](#using-ivy) further down this page 👇 which contain a number of models and libraries transpiled between PyTorch, JAX, TensorFlow and NumPy.

<br>

# Using ivy

Here's some examples, to help you get started using Ivy! The [examples page](https://ivy.dev/docs/demos/) also features a wide range of
demos and tutorials showcasing some more use cases for Ivy.

  <details>
   <summary><b>Transpiling any code from one framework to another</b></summary>

   ``` python
   import ivy
   import torch
   import tensorflow as tf

   def torch_fn(x):
       a = torch.mul(x, x)
       b = torch.mean(x)
       return x * a + b

   tf_fn = ivy.transpile(torch_fn, source="torch", target="tensorflow")

   tf_x = tf.convert_to_tensor([1., 2., 3.])
   ret = tf_fn(tf_x)
   ```

   </details>

  <details>
   <summary><b>Tracing a computational graph of any code</b></summary>

   ``` python
   import ivy
   import torch

   def torch_fn(x):
       a = torch.mul(x, x)
       b = torch.mean(x)
       return x * a + b

   torch_x = torch.tensor([1., 2., 3.])
   graph = ivy.trace_graph(jax_fn, to="torch", args=(torch_x,))
   ret = graph(torch_x)
   ```

   </details>

<!-- <details>
<summary><b>I'm using PyTorch&ensp;<img class="dark-light" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/torch_small_logo.png"></b></summary>
   <blockquote>You can use Ivy to get PyTorch code from:
      <details>
         <summary>Any model</summary>
         <blockquote>
            <details>
               <summary>From TensorFlow</summary>

``` python
import ivy
import torch
import tensorflow as tf

# Get a pretrained keras model
eff_encoder = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)

# Transpile it into a torch.nn.Module with the corresponding parameters
noise = tf.random.normal(shape=(1, 224, 224, 3))
torch_eff_encoder = ivy.transpile(eff_encoder, source="tensorflow", to="torch", args=(noise,))

# Build a classifier using the transpiled encoder
class Classifier(torch.nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.encoder = torch_eff_encoder
        self.fc = torch.nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)

# Initialize a trainable, customizable, torch.nn.Module
classifier = Classifier()
ret = classifier(torch.rand((1, 244, 244, 3)))
```

</details>
<details>
   <summary>From JAX</summary>

``` python
import ivy
import jax
import torch

# Get a pretrained haiku model
# https://github.com/unifyai/demos/blob/15c235f/scripts/deepmind_perceiver_io.py
from deepmind_perceiver_io import key, perceiver_backbone

# Transpile it into a torch.nn.Module with the corresponding parameters
dummy_input = jax.random.uniform(key, shape=(1, 3, 224, 224))
params = perceiver_backbone.init(rng=key, images=dummy_input)
ivy.set_backend("jax")
backbone = ivy.transpile(
    perceiver_backbone, source="jax", to="torch", params_v=params, kwargs={"images": dummy_input}
)

# Build a classifier using the transpiled backbone
class PerceiverIOClassifier(torch.nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = backbone
        self.max_pool = torch.nn.MaxPool2d((512, 1))
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(images=x)
        x = self.flatten(self.max_pool(x))
        return self.fc(x)

# Initialize a trainable, customizable, torch.nn.Module
classifier = PerceiverIOClassifier()
ret = classifier(torch.rand((1, 3, 224, 224)))
```

</details>
</blockquote>
</details>

<details>
<summary>Any library</summary>
<blockquote>
<details>
   <summary>From Tensorflow</summary>

``` python
import ivy
import torch
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

# transpile sm from tensorflow to torch
torch_sm = ivy.transpile(sm, source="tensorflow", to="torch")

# get some image-like arrays
output = torch.rand((1, 3, 512, 512))
target = torch.rand((1, 3, 512, 512))

# and use the transpiled version of any function from the library!
out = torch_sm.metrics.iou_score(output, target)
```

</details>
<details>
   <summary>From JAX</summary>

``` python
import ivy
import rax
import torch

# transpile rax from jax to torch
torch_rax = ivy.transpile(rax, source="jax", to="torch")

# get some arrays
scores = torch.tensor([2.2, 1.3, 5.4])
labels = torch.tensor([1.0, 0.0, 0.0])

# and use the transpiled version of any function from the library!
out = torch_rax.poly1_softmax_loss(scores, labels)
```

</details>
<details>
   <summary>From NumPy</summary>

``` python
import ivy
import torch
import madmom

# transpile madmon from numpy to torch
torch_madmom = ivy.transpile(madmom, source="numpy", to="torch")

# get some arrays
freqs = torch.arange(20) * 10

# and use the transpiled version of any function from the library!
out = torch_madmom.audio.filters.hz2midi(freqs)
```

</details>
</blockquote>
</details>

<details>
<summary>Any function</summary>
<blockquote>
<details>
   <summary>From Tensorflow</summary>

``` python
import ivy
import tensorflow as tf
import torch

def loss(predictions, targets):
    return tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

# transpile any function from tf to torch
torch_loss = ivy.transpile(loss, source="tensorflow", to="torch")

# get some arrays
p = torch.tensor([3.0, 2.0, 1.0])
t = torch.tensor([0.0, 0.0, 0.0])

# and use the transpiled version!
out = torch_loss(p, t)
```

</details>
<details>
   <summary>From JAX</summary>

``` python
import ivy
import jax.numpy as jnp
import torch

def loss(predictions, targets):
    return jnp.sqrt(jnp.mean((predictions - targets) ** 2))

# transpile any function from jax to torch
torch_loss = ivy.transpile(loss, source="jax", to="torch")

# get some arrays
p = torch.tensor([3.0, 2.0, 1.0])
t = torch.tensor([0.0, 0.0, 0.0])

# and use the transpiled version!
out = torch_loss(p, t)
```

</details>
<details>
   <summary>From NumPy</summary>

``` python
import ivy
import numpy as np
import torch

def loss(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

# transpile any function from numpy to torch
torch_loss = ivy.transpile(loss, source="numpy", to="torch")

# get some arrays
p = torch.tensor([3.0, 2.0, 1.0])
t = torch.tensor([0.0, 0.0, 0.0])

# and use the transpiled version!
out = torch_loss(p, t)
```

</details>
</blockquote>
</details>

</blockquote>
</details>

<details>
<summary><b>I'm using TensorFlow&ensp;<img class="dark-light" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/tf_small_logo.png"></b></summary>
<blockquote>You can use Ivy to get TensorFlow code from:
<details>
<summary>Any model</summary>
<blockquote>
<details>
   <summary>From PyTorch</summary>

``` python
import ivy
import torch
import timm
import tensorflow as tf

# Get a pretrained pytorch model
mlp_encoder = timm.create_model("mixer_b16_224", pretrained=True, num_classes=0)

# Transpile it into a keras.Model with the corresponding parameters
noise = torch.randn(1, 3, 224, 224)
mlp_encoder = ivy.transpile(mlp_encoder, to="tensorflow", args=(noise,))

# Build a classifier using the transpiled encoder
class Classifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = mlp_encoder
        self.output_dense = tf.keras.layers.Dense(units=1000, activation="softmax")

    def call(self, x):
        x = self.encoder(x)
        return self.output_dense(x)

# Transform the classifier and use it as a standard keras.Model
x = tf.random.normal(shape=(1, 3, 224, 224))
model = Classifier()
ret = model(x)
```

</details>
<details>
   <summary>From JAX</summary>

``` python
import ivy
import jax
import tensorflow as tf

# Get a pretrained haiku model
# https://ivy.dev/demos/scripts/deepmind_perceiver_io.py
from deepmind_perceiver_io import key, perceiver_backbone

# Transpile it into a tf.keras.Model with the corresponding parameters
dummy_input = jax.random.uniform(key, shape=(1, 3, 224, 224))
params = perceiver_backbone.init(rng=key, images=dummy_input)
backbone = ivy.transpile(
    perceiver_backbone, to="tensorflow", params_v=params, args=(dummy_input,)
)

# Build a classifier using the transpiled backbone
class PerceiverIOClassifier(tf.keras.Model):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = backbone
        self.max_pool = tf.keras.layers.MaxPooling1D(pool_size=512)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.backbone(x)
        x = self.flatten(self.max_pool(x))
        return self.fc(x)

# Initialize a trainable, customizable, tf.keras.Model
x = tf.random.normal(shape=(1, 3, 224, 224))
classifier = PerceiverIOClassifier()
ret = classifier(x)
```

</details>
</blockquote>
</details>

<details>
<summary>Any library</summary>
<blockquote>
<details>
   <summary>From PyTorch</summary>

``` python
import ivy
import kornia
import requests
import numpy as np
import tensorflow as tf
from PIL import Image

# transpile kornia from torch to tensorflow
tf_kornia = ivy.transpile(kornia, source="torch", to="tensorflow")

# get an image
url = "http://images.cocodataset.org/train2017/000000000034.jpg"
raw_img = Image.open(requests.get(url, stream=True).raw)

# convert it to the format expected by kornia
img = np.array(raw_img)
img = tf.transpose(tf.constant(img), (2, 0, 1))
img = tf.expand_dims(img, 0) / 255

# and use the transpiled version of any function from the library!
out = tf_kornia.enhance.sharpness(img, 5)
```

</details>
<details>
   <summary>From JAX</summary>

``` python
import ivy
import rax
import tensorflow as tf

# transpile rax from jax to tensorflow
tf_rax = ivy.transpile(rax, source="jax", to="tensorflow")

# get some arrays
scores = tf.constant([2.2, 1.3, 5.4])
labels = tf.constant([1.0, 0.0, 0.0])

# and use the transpiled version of any function from the library!
out = tf_rax.poly1_softmax_loss(scores, labels)
```

</details>
<details>
   <summary>From NumPy</summary>

``` python
import ivy
import madmom
import tensorflow as tf

# transpile madmom from numpy to tensorflow
tf_madmom = ivy.transpile(madmom, source="numpy", to="tensorflow")

# get some arrays
freqs = tf.range(20) * 10

# and use the transpiled version of any function from the library!
out = tf_madmom.audio.filters.hz2midi(freqs)
```

</details>
</blockquote>
</details>

<details>
<summary>Any function</summary>
<blockquote>
<details>
   <summary>From PyTorch</summary>

``` python
import ivy
import torch
import tensorflow as tf

def loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# transpile any function from torch to tensorflow
tf_loss = ivy.transpile(loss, source="torch", to="tensorflow")

# get some arrays
p = tf.constant([3.0, 2.0, 1.0])
t = tf.constant([0.0, 0.0, 0.0])

# and use the transpiled version!
out = tf_loss(p, t)
```

</details>
<details>
   <summary>From JAX</summary>

``` python
import ivy
import jax.numpy as jnp
import tensorflow as tf

def loss(predictions, targets):
    return jnp.sqrt(jnp.mean((predictions - targets) ** 2))

# transpile any function from jax to tensorflow
tf_loss = ivy.transpile(loss, source="jax", to="tensorflow")

# get some arrays
p = tf.constant([3.0, 2.0, 1.0])
t = tf.constant([0.0, 0.0, 0.0])

# and use the transpiled version!
out = tf_loss(p, t)
```

</details>
<details>
   <summary>From NumPy</summary>

``` python
import ivy
import numpy as np
import tensorflow as tf

def loss(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

# transpile any function from numpy to tensorflow
tf_loss = ivy.transpile(loss, source="numpy", to="tensorflow")

# get some arrays
p = tf.constant([3.0, 2.0, 1.0])
t = tf.constant([0.0, 0.0, 0.0])

# and use the transpiled version!
out = tf_loss(p, t)
```

</details>
</blockquote>
</details>

</blockquote>
</details>

<details>
<summary><b>I'm using Jax&ensp;<img class="dark-light" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/jax_small_logo.png"></b></summary>
<blockquote>You can use Ivy to get JAX code from:
<details>
<summary>Any model</summary>
<blockquote>
<details>
   <summary>From PyTorch</summary>

``` python
import ivy
import timm
import torch
import jax
import haiku as hk

# Get a pretrained pytorch model
mlp_encoder = timm.create_model("mixer_b16_224", pretrained=True, num_classes=0)

# Transpile it into a hk.Module with the corresponding parameters
noise = torch.randn(1, 3, 224, 224)
mlp_encoder = ivy.transpile(mlp_encoder, source="torch", to="haiku", args=(noise,))

# Build a classifier using the transpiled encoder
class Classifier(hk.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.encoder = mlp_encoder()
        self.fc = hk.Linear(output_size=num_classes, with_bias=True)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

def _forward_classifier(x):
    module = Classifier()
    return module(x)

# Transform the classifier and use it as a standard hk.Module
rng_key = jax.random.PRNGKey(42)
x = jax.random.uniform(key=rng_key, shape=(1, 3, 224, 224), dtype=jax.numpy.float32)
forward_classifier = hk.transform(_forward_classifier)
params = forward_classifier.init(rng=rng_key, x=x)

ret = forward_classifier.apply(params, None, x)
```

</details>
<details>
   <summary>From TensorFlow</summary>

``` python
import ivy
import jax
import haiku as hk
import tensorflow as tf
jax.config.update("jax_enable_x64", True)

# Get a pretrained keras model
eff_encoder = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)

# Transpile it into a hk.Module with the corresponding parameters
noise = tf.random.normal(shape=(1, 224, 224, 3))
hk_eff_encoder = ivy.transpile(eff_encoder, source="tensorflow", to="haiku", args=(noise,))

# Build a classifier using the transpiled encoder
class Classifier(hk.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.encoder = hk_eff_encoder()
        self.fc = hk.Linear(output_size=num_classes, with_bias=True)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

def _forward_classifier(x):
    module = Classifier()
    return module(x)

# Transform the classifier and use it as a standard hk.Module
rng_key = jax.random.PRNGKey(42)
dummy_x = jax.random.uniform(key=rng_key, shape=(1, 224, 224, 3))
forward_classifier = hk.transform(_forward_classifier)
params = forward_classifier.init(rng=rng_key, x=dummy_x)

ret = forward_classifier.apply(params, None, dummy_x)
```

</details>
</blockquote>
</details>

<details>
<summary>Any library</summary>
<blockquote>
<details>
   <summary>From PyTorch</summary>

``` python
import ivy
import kornia
import requests
import jax.numpy as jnp
from PIL import Image
jax.config.update("jax_enable_x64", True)

# transpile kornia from torch to jax
jax_kornia = ivy.transpile(kornia, source="torch", to="jax")

# get an image
url = "http://images.cocodataset.org/train2017/000000000034.jpg"
raw_img = Image.open(requests.get(url, stream=True).raw)

# convert it to the format expected by kornia
img = jnp.transpose(jnp.array(raw_img), (2, 0, 1))
img = jnp.expand_dims(img, 0) / 255

# and use the transpiled version of any function from the library!
out = jax_kornia.enhance.sharpness(img, 5)
```

</details>
<details>
   <summary>From TensorFlow</summary>

``` python
import ivy
import jax
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

# transpile sm from tensorflow to jax
jax_sm = ivy.transpile(sm, source="tensorflow", to="jax")

# get some image-like arrays
key = jax.random.PRNGKey(23)
key1, key2 = jax.random.split(key)
output = jax.random.uniform(key1, (1, 3, 512, 512))
target = jax.random.uniform(key2, (1, 3, 512, 512))

# and use the transpiled version of any function from the library!
out = jax_sm.metrics.iou_score(output, target)
```

</details>
<details>
   <summary>From NumPy</summary>

``` python
import ivy
import madmom
import jax.numpy as jnp

# transpile madmon from numpy to jax
jax_madmom = ivy.transpile(madmom, source="numpy", to="jax")

# get some arrays
freqs = jnp.arange(20) * 10

# and use the transpiled version of any function from the library!
out = jax_madmom.audio.filters.hz2midi(freqs)
```

</details>
</blockquote>
</details>

<details>
<summary>Any function</summary>
<blockquote>
<details>
   <summary>From PyTorch</summary>

``` python
import ivy
import torch
import jax.numpy as jnp

def loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# transpile any function from torch to jax
jax_loss = ivy.transpile(loss, source="torch", to="jax")

# get some arrays
p = jnp.array([3.0, 2.0, 1.0])
t = jnp.array([0.0, 0.0, 0.0])

# and use the transpiled version!
out = jax_loss(p, t)
```

</details>
<details>
   <summary>From TensorFlow</summary>

``` python
import ivy
import tensorflow as tf
import jax.numpy as jnp

def loss(predictions, targets):
    return tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

# transpile any function from tf to jax
jax_loss = ivy.transpile(loss, source="tensorflow", to="jax")

# get some arrays
p = jnp.array([3.0, 2.0, 1.0])
t = jnp.array([0.0, 0.0, 0.0])

# and use the transpiled version!
out = jax_loss(p, t)
```

</details>
<details>
   <summary>From NumPy</summary>

``` python
import ivy
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

def loss(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

# transpile any function from numpy to jax
jax_loss = ivy.transpile(loss, source="numpy", to="jax")

# get some arrays
p = jnp.array([3.0, 2.0, 1.0])
t = jnp.array([0.0, 0.0, 0.0])

# and use the transpiled version!
out = jax_loss(p, t)
```

</details>
</blockquote>
</details>

</blockquote>
</details>

<details>
<summary><b>I'm using NumPy&ensp;<img class="dark-light" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/numpy_small_logo.png"></b></summary>
<blockquote>You can use Ivy to get NumPy code from:
<details>
<summary>Any library</summary>
<blockquote>
<details>
   <summary>From PyTorch</summary>

``` python
import ivy
import kornia
import requests
import numpy as np
from PIL import Image

# transpile kornia from torch to np
np_kornia = ivy.transpile(kornia, source="torch", to="numpy")

# get an image
url = "http://images.cocodataset.org/train2017/000000000034.jpg"
raw_img = Image.open(requests.get(url, stream=True).raw)

# convert it to the format expected by kornia
img = np.transpose(np.array(raw_img), (2, 0, 1))
img = np.expand_dims(img, 0) / 255

# and use the transpiled version of any function from the library!
out = np_kornia.enhance.sharpness(img, 5)
```

</details>
<details>
   <summary>From TensorFlow</summary>

``` python
import ivy
import numpy as np
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

# transpile sm from tensorflow to numpy
np_sm = ivy.transpile(sm, source="tensorflow", to="numpy")

# get some image-like arrays
output = np.random.rand(1, 3, 512, 512).astype(dtype=np.float32)
target = np.random.rand(1, 3, 512, 512).astype(dtype=np.float32)

# and use the transpiled version of any function from the library!
out = np_sm.metrics.iou_score(output, target)
```

</details>
<details>
   <summary>From Jax</summary>

``` python
import ivy
import rax
import numpy as np

# transpile rax from jax to numpy
np_rax = ivy.transpile(rax, source="jax", to="numpy")

# get some arrays
scores = np.array([2.2, 1.3, 5.4])
labels = np.array([1.0, 0.0, 0.0])

# and use the transpiled version of any function from the library!
out = np_rax.poly1_softmax_loss(scores, labels)
```

</details>
</blockquote>
</details>

<details>
<summary>Any function</summary>
<blockquote>
<details>
   <summary>From PyTorch</summary>

``` python
import ivy
import torch
import numpy as np

def loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# transpile any function from torch to numpy
np_loss = ivy.transpile(loss, source="torch", to="numpy")

# get some arrays
p = np.array([3.0, 2.0, 1.0])
t = np.array([0.0, 0.0, 0.0])

# and use the transpiled version!
out = np_loss(p, t)
```

</details>
<details>
   <summary>From TensorFlow</summary>

``` python
import ivy
import tensorflow as tf
import numpy as np

def loss(predictions, targets):
    return tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

# transpile any function from tf to numpy
np_loss = ivy.transpile(loss, source="tensorflow", to="numpy")

# get some arrays
p = np.array([3.0, 2.0, 1.0])
t = np.array([0.0, 0.0, 0.0])

# and use the transpiled version!
out = np_loss(p, t)
```

</details>
<details>
   <summary>From JAX</summary>

``` python
import ivy
import jax.numpy as jnp
import numpy as np

def loss(predictions, targets):
    return jnp.sqrt(jnp.mean((predictions - targets) ** 2))

# transpile any function from jax to numpy
np_loss = ivy.transpile(loss, source="jax", to="numpy")

# get some arrays
p = np.array([3.0, 2.0, 1.0])
t = np.array([0.0, 0.0, 0.0])

# and use the transpiled version!
out = np_loss(p, t)
```

</details>
</blockquote>
</details>

</blockquote>
</details> -->

<br>

# How ivy works?

Let's take a look at how Ivy works as a transpiler in more detail to get an idea of why and where to use it.

<blockquote>
<details>
<summary>When is Ivy's transpiler useful?</summary>

If you want to use building blocks published in other frameworks (neural
networks, layers, array computing libraries, training pipelines\...),
you want to integrate code developed in various frameworks, or maybe
straight up migrate code from one framework to another or even between versions of the same framework, the transpiler is
definitely the tool for the job! You can use the converted code just
as if it was code originally developed in that framework, applying
framework-specific optimizations or tools, instantly exposing your
project to all of the unique perks of a different framework.
</details>
</blockquote>

\
Ivy\'s transpiler allows you to use code from any other framework (or
from any other version of the same framework!) in your own code, by just
adding one line of code.

This way, Ivy makes all ML-related projects available for you,
independently of the framework you want to use to research, develop, or
deploy systems. Feel free to head over to the docs for the full API
reference, but the functions you\'d most likely want to use are:

``` python
# Converts framework-specific code to a target framework of choice. See usage in the documentation
ivy.transpile()

# Traces an efficient fully-functional graph from a function, removing all wrapping and redundant code. See usage in the documentation
ivy.trace_graph()
```

#### `ivy.transpile` will eagerly transpile if a class or function is provided

``` python
import ivy
import torch
import tensorflow as tf

def torch_fn(x):
    x = torch.abs(x)
    return torch.sum(x)

x1 = torch.tensor([1., 2.])
x1 = tf.convert_to_tensor([1., 2.])

# Transpilation happens eagerly
tf_fn = ivy.transpile(test_fn, source="torch", target="tensorflow")

# tf_fn is now tensorflow code and runs efficiently
ret = tf_fn(x1)
```

#### `ivy.transpile` will lazily transpile if a module (library) is provided

``` python
import kornia

x2 = torch.rand(5, 3, 4, 4)

# Module is provided -> transpilation happens lazily
tf_kornia = ivy.transpile(kornia, source="torch", target="tensorflow")

# The transpilation is initialized here, and this function is converted to tensorflwo
ret = tf_kornia.color.rgb_to_grayscale(x2)

# Transpilation has already occurred, the tensorflow function runs efficiently
ret = tf_kornia.color.rgb_to_grayscale(x2)
```

#### `ivy.trace_graph` can be used eagerly or lazily
If you pass the necessary arguments for function tracing, the graph tracing step will
happen instantly (eagerly). Otherwise, the graph tracing
will happen only when the returned function is first invoked.

``` python
import ivy
import jax
ivy.set_backend("jax")

# Simple JAX function to transpile
def test_fn(x):
    return jax.numpy.sum(x)

x1 = ivy.array([1., 2.])
```

``` python
# Arguments are available -> tracing happens eagerly
eager_graph = ivy.trace_graph(test_fn, to="jax", args=(x1,))

# eager_graph now runs efficiently
ret = eager_graph(x1)
```

``` python
# Arguments are not available -> tracing happens lazily
lazy_graph = ivy.trace_graph(test_fn, to="jax")

# The traced graph is initialized, tracing will happen here
ret = lazy_graph(x1)

# Tracing has already happend, traced graph runs efficiently
ret = lazy_graph(x1)
```

If you want to learn more, you can find more information in the [Ivy as
a transpiler section of the
docs!](https://ivy.dev/docs/overview/design/ivy_as_a_transpiler.html)


<br>

# Documentation

You can find Ivy's documentation on the [Docs page](https://ivy.dev/docs/), which includes:
- [Motivation](https://ivy.dev/docs/overview/motivation.html): This contextualizes the problem Ivy is trying to solve by going over
    - The current [ML Explosion](https://ivy.dev/docs/overview/motivation/ml_explosion.html#ml-explosion).
    - Explaining why it is important [to solve this problem](https://ivy.dev/docs/overview/motivation/why_unify.html#why-unify).
    - Explaining how we adhere to existing [standards](https://ivy.dev/docs/overview/motivation/standardization.html#standardization) to make this happen.
- [Related Work](https://ivy.dev/docs/overview/related_work.html): Which paints a picture of the role Ivy plays in the ML stack, comparing it to other existing solutions in terms of functionalities and abstraction level.
- [Design](https://ivy.dev/docs/overview/design.html): A user-focused guide about the design decision behind the architecture and the main building blocks of Ivy.
- [Deep Dive](https://ivy.dev/docs/overview/deep_dive.html): Which delves deeper into the implementation details of Ivy and is oriented towards potential contributors to the code base.

<br>

# Contributing


We believe that everyone can contribute and make a difference. Whether
it\'s writing code, fixing bugs, or simply sharing feedback,
your contributions are definitely welcome and appreciated 🙌

Check out all of our [Open Tasks](https://ivy.dev/docs/overview/contributing/open_tasks.html),
and find out more info in our [Contributing guide](https://ivy.dev/docs/overview/contributing.html)
in the docs! Or to immediately dive into a useful task, look for any failing tests on our [Test Dashboard](https://github.com/ivy-llc/ivy-tests-dashboard/blob/main/DASHBOARD.md)!

<a href="https://github.com/ivy-llc/ivy/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=ivy-llc/ivy&anon=0&columns=20&max=100&r=true" />
</a>

<br>

# Community


Join our growing community on a mission to make conversions between frameworks simple and accessible to all!
Whether you are a seasoned developer or just starting out, you\'ll find a place here! Join the Ivy community on
our [Discord](https://discord.gg/mMnS8Egy) 👾 server, which is the
perfect place to ask questions, share ideas, and get help from both
fellow developers and the Ivy Team directly.

<b> See you there! </b>


<br>

# Citation

If you use Ivy for your work, please don\'t forget to give proper credit
by including the accompanying [paper](https://arxiv.org/abs/2102.02886)
📄 in your references. It\'s a small way to show appreciation and help
to continue to support this and other open source projects 🙌


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
