.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo.png?raw=true
   :width: 100%

.. raw:: html

    <br/>
    <div align="center">
    <a href="https://github.com/unifyai/ivy/issues">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/issues/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/network/members">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/forks/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/stargazers">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/pulls">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
    </a>
    <a href="https://pypi.org/project/ivy-core">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-core.svg">
    </a>
    <a href="https://github.com/unifyai/ivy/actions?query=workflow%3Adocs">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/ivy/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://github.com/unifyai/ivy/actions?query=workflow%3Atest-ivy">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/ivy/actions/workflows/test-ivy.yml/badge.svg">
    </a>
    <a href="https://discord.gg/sXyFF8tDtm">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    </div>
    <br clear="all" />

    <h4 align="center">
        <p>
            <a href="https://github.com/unifyai/ivy">English</a> |
            <b>한국어</b>

        <p>
    </h4>

**Ivy와 함께 모든 ML framework를 통합하고 💥 + 자동으로 코드 변환까지 진행해보세요 🔄.**

**pip install ivy-core 이후 🚀 Ivy팀의 성장하는 community에 가입하시고 😊, 통합된 환경을 구축하세요! 🦾**

.. raw:: html

    <div style="display: block;" align="center">
        <img width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
        <img width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
    </div>


.. _docs: https://lets-unify.ai/ivy
.. _Colabs: https://drive.google.com/drive/folders/16Oeu25GrQsEJh8w2B0kSrD93w4cWjJAM?usp=sharing
.. _`contributor guide`: https://lets-unify.ai/ivy/contributing.html
.. _`open tasks`: https://lets-unify.ai/ivy/contributing/open_tasks.html


Contents
--------

* `Ivy란?`_
* `시작하기`_
* `Background`_
* `Design`_
* `Extensions`_
* `Contributing`_

Ivy란?
--------
Ivy는 JAX, TensorFlow, PyTorch 및 Numpy를 지원하는 ML 프레임워크입니다.

IVY의 다음 목표는 모든 프레임워크 간의 자동 코드 변환을 지원하고,
모든 오픈 소스 라이브러리에 대해 단 몇 줄의 코드만 변경함으로써 다양한 프레임워크를 지원하는 것입니다.
더 많은 정보를 알아보려면 아래를 참조하세요.😊

문서는 Ivy를 왜 만들었는지, 어떻게 사용하는지, 우리의 로드맵에서 무엇을 계획하고 있는지와 
contribute하는 방법에 대해 다룬 sub-page로 구성되어 있습니다.
Contents의 각 항목을 클릭하시면 sub-page 조회가 가능합니다.

현재 개발 중인 기능은 🚧, 이미 구현된 기능에 대해서는 ✅로 표시합니다.

더 많은 정보를 원하시면 docs_ 를 참고해주시고,
예제 코드는 Google Colabs_ 를 참고해주세요!


🚨 Ivy는 아직 상대적으로 개발 초기 단계입니다. 앞으로 몇 주 안에 버전 1.2.0을 출시할 때까지 획기적인 변화를 기대해주세요!

만약 contribute하는 것을 원하시면, `contributor guide`_ 와 `open tasks`_ 를 참고해주세요 🧑‍💻

시작하기
-----------

Ivy는 ``pip install ivy-core`` 로 설치할 수 있습니다.
아래와 같이, 사용자가 선호하는 프레임워크를 background에서 선택하여 신경망을 훈련시킬 수 있습니다

.. code-block:: python

    import ivy

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))

    ivy.set_backend('torch')  # change to any backend!
    model = MyModel()
    optimizer = ivy.Adam(1e-4)
    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.mean((out - target)**2)

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
        print('step {} loss {}'.format(step, ivy.to_numpy(loss).item()))

    print('Finished training!')

이 예제는 backend framework로 PyTorch를 사용하였습니다.
backend는 TensorFlow, JAX와 같은 사용자가 선호하는 프레임워크로 쉽게 변경이 가능합니다.

**Framework Agnostic Functions**

아래의 예제에서는 다양한 프레임워크의 tensor와 호환되는 Ivy의 concatenation 함수를 사용하였습니다.
이는 Ivy의 모든 함수에 적용됩니다. 모든 Ivy 함수는 어떤 프레임워크에서든 tensor를 받아들이고, 결과를 반환합니다.

.. code-block:: python

    import jax.numpy as jnp
    import tensorflow as tf
    import numpy as np
    import torch

    import ivy

    jax_concatted   = ivy.concat((jnp.ones((1,)), jnp.ones((1,))), -1)
    tf_concatted    = ivy.concat((tf.ones((1,)), tf.ones((1,))), -1)
    np_concatted    = ivy.concat((np.ones((1,)), np.ones((1,))), -1)
    torch_concatted = ivy.concat((torch.ones((1,)), torch.ones((1,))), -1)

Ivy의 모든 method들을 살펴보려면, python command prompt에서 :code:`ivy.` 를 입력하고 :code:`tab` 을 누르세요.
결과는 다음과 같습니다.

::

   ivy.Container(                         ivy.general                               ivy.reduce_min(
   ivy.abs(                               ivy.get_device(                           ivy.reduce_prod(
   ivy.acos(                              ivy.get_num_dims(                         ivy.reduce_sum(
   ivy.acosh(                             ivy.gradient_descent_update(              ivy.reductions
   ivy.activations                        ivy.gradient_image(                       ivy.relu(
   ivy.arange(                            ivy.gradients                             ivy.reshape(
   ivy.argmax(                            ivy.identity(                             ivy.round(
   ivy.argmin(                            ivy.image                                 ivy.scatter_nd(
   ivy.array(                             ivy.indices_where(                        ivy.seed(
   ivy.asin(                              ivy.inv(                                  ivy.shape(
   ivy.asinh(                             ivy.layers                                ivy.shuffle(
   ivy.atan(                              ivy.leaky_relu(                           ivy.sigmoid(
   ivy.atan2(                             ivy.linalg                                ivy.sin(
   ivy.atanh(                             ivy.linear(                               ivy.sinh(
   ivy.bilinear_resample(                 ivy.linspace(                             ivy.softmax(
   ivy.cast(                              ivy.log(                                  ivy.softplus(
   ivy.ceil(                              ivy.logic                                 ivy.split(
   ivy.clip(                              ivy.logical_and(                          ivy.squeeze(
   ivy.concatenate(                       ivy.logical_not(                          ivy.stack(            
   ivy.container                          ivy.logical_or(                           ivy.stack_images(
   ivy.conv2d(                            ivy.math                                  ivy.stop_gradient(
   ivy.core                               ivy.matmul(                               ivy.svd(
   ivy.cos(                               ivy.maximum(                              ivy.tan(
   ivy.cosh(                              ivy.minimum(                              ivy.tanh(
   ivy.cross(                             ivy.neural_net                            ivy.tile(
   ivy.cumsum(                            ivy.nn                                    ivy.to_list(
   ivy.depthwise_conv2d(                  ivy.norm(                                 ivy.to_numpy(
   ivy.dtype(                             ivy.one_hot(                              ivy.transpose(
   ivy.execute_with_gradients(            ivy.ones(                                 ivy.unstack(
   ivy.exp(                               ivy.ones_like(                            ivy.vector_norm(
   ivy.expand_dims(                       ivy.pinv(                                 ivy.vector_to_skew_symmetric_matrix(
   ivy.flip(                              ivy.randint(                              ivy.verbosity
   ivy.floor(                             ivy.random                                ivy.where(
   ivy.floormod(                          ivy.random_uniform(                       ivy.zero_pad(
   ivy.backend_handler                    ivy.reduce_max(                           ivy.zeros(
   ivy.gather_nd(                         ivy.reduce_mean(                          ivy.zeros_like(

Background
----------

| (a) `ML Explosion <https://lets-unify.ai/ivy/background/ml_explosion.html>`_
| 많은 ML framework들이 등장하고 있습니다.
|
| (b) `Why Unify? <https://lets-unify.ai/ivy/background/why_unify.html>`_
| 왜 ML framework들을 통합해야 할까요?
|
| (c) `Standardization <https://lets-unify.ai/ivy/background/standardization.html>`_
| Ivy는 `Consortium for Python Data API Standards <https://data-apis.org>`_ 와 협력합니다.

Design
------

| Ivy는 두 가지의 역할을 수행할 수 있습니다:
| 
| 1. Framework간 transpiler 역할 수행 🚧
| 2. Multi-framework 지원을 통한 새로운 ML framework 역할 수행 ✅
|
| Ivy의 codebase는 세 가지의 카테고리로 나눌 수 있으며, 8개의 distinct한 submodule로 나눌 수 있습니다. 각각은 다음과 같은 세 가지 카테고리 중 하나에 속합니다

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

| (a) `Block 구성하기 <https://lets-unify.ai/ivy/design/building_blocks.html>`_
| Backend functional APIs ✅
| Ivy functional API ✅
| Backend Handler ✅
| Ivy Compiler 🚧
|
| (b) `Transpiler로서의 Ivy <https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html>`_
| Front-end functional APIs 🚧
|
| (c) `Framework로서의 Ivy <https://lets-unify.ai/ivy/design/ivy_as_a_framework.html>`_
| Ivy stateful API ✅
| Ivy Container ✅
| Ivy Array 🚧

Extensions
----------

| (a) `Applied Libraries <https://lets-unify.ai/ivy/extensions/applied_libraries.html>`_ ✅
| mechanics, vision, robotics, memory 및 다른 기타 분야에 적용할 수 있는 Ivy library들입니다.
|
| (b) **Builder [Docs 제작 중입니다!]** ✅
| 단 몇 줄의 코드만으로 학습 workflow를 구성하는데 도움이 되는 :code:`ivy.Trainer`, :code:`ivy.Dataset`, :code:`ivy.Dataloader` 및 기타 class들입니다.

Contributing
------------

Ivy community에 code contributor로 합류하시고, 모든 ML Framework를 통합하는 것을 도와주세요!
저희의 모든 open task를 확인하시고, `Contributing <https://lets-unify.ai/ivy/contributing.html>`_ 가이드에서 더 많은 정보를 확인하세요!

Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
