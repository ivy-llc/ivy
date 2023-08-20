Standardization
===============

Skepticism
----------

With our central goal being to unify all ML frameworks, you would be entirely forgiven for raising an eyebrow 🤨

“You want to try and somehow unify: TensorFlow, PyTorch, JAX, NumPy and others, all of which have strong industrial backing, huge user momentum, and significant API differences?”

Won’t adding a new “unified” framework just make the problem even worse…

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/background/standardization/how_standards_proliferate.png?raw=true
   :align: center
   :width: 70%
   :target: https://xkcd.com/927/

Complimentary vs Competitive
----------------------------


When considering this rather funny comic strip, it’s easy to question the feasibility of standardization in this ML space.
However, there is a very important difference in the case of Ivy.
Unlike `A/C Chargers <https://en.wikipedia.org/wiki/AC_adapter#Problems>`_, `Character Encodings <https://en.wikipedia.org/wiki/Character_encoding>`_ and `Instant Messaging <https://en.wikipedia.org/wiki/Comparison_of_instant_messaging_protocols>`_ where it’s very much one standard **or** the other, with Ivy this is not the case.

While Ivy does adhere to the `Python Array API Standard <https://data-apis.org/array-api/latest/>`_, **Ivy does not mandate adoption of the standard**.
Ivy is entirely complimentary to existing frameworks, each of which can and will continue to have their own syntax and call signatures.
**Ivy is not a replacement**.
Your project can have 1% Ivy code, 10% Ivy code, or 100% Ivy code.
This is entirely up to you!

Do Standards Work?
------------------

Despite making this distinction with Ivy, we do still believe that the adoption of a shared standard by each ML framework would bring huge benefits, unrelated to what we’re doing at Ivy.

Again, contrary to `A/C Chargers <https://en.wikipedia.org/wiki/AC_adapter#Problems>`_, `Character Encodings <https://en.wikipedia.org/wiki/Character_encoding>`_, `Instant Messaging <https://en.wikipedia.org/wiki/Comparison_of_instant_messaging_protocols>`_ and other bumpy roads alluded to in the comic, most of the technology sector is full of successful standards.
The reason we can “build” custom computers is thanks to many essential standards for the interoperability of different computer components, such as: `BIOS <https://en.wikipedia.org/wiki/BIOS#BIOS_Boot_Specification>`_ for hardware initialization, `PCIe <https://en.wikipedia.org/wiki/PCI_Express>`_ for interfacing components on the motherboard, `RAID <https://en.wikipedia.org/wiki/RAID>`_ for storage virtualization, `Bluetooth <https://en.wikipedia.org/wiki/Bluetooth>`_ for wireless data exchange, `BTX <https://en.wikipedia.org/wiki/BTX_(form_factor)>`_ for motherboard form factors and `SATA <https://en.wikipedia.org/wiki/Serial_ATA>`_ for connecting host bus adapters to storage devices.

For software, `HTML <https://en.wikipedia.org/wiki/HTML>`_ enables anyone to design and host a website, `TCP/IP <https://en.wikipedia.org/wiki/Internet_protocol_suite#>`_ enables different nodes to communicate on a network, `SMTP <https://en.wikipedia.org/wiki/Simple_Mail_Transfer_Protocol>`_ makes it possible to send from Gmail to Outlook, `POP <https://en.wikipedia.org/wiki/Post_Office_Protocol>`_ enables us to open this email and `IEEE 754 <https://en.wikipedia.org/wiki/IEEE_754>`_ allows us to do calculations.
These are all essential standards which our modern lives depend on.
Most of these standards did not arise until there was substantial innovation, growth and usage and in the relevant area, making standardization a necessity so that all parties could easily engage.

With regards to array libraries in Python, NumPy was effectively the standard until ~2015.
Since then, array libraries have seen an explosion alongside innovations in Machine Learning.
Given this recent time-frame, we are in a much less mature state than all of the preceding standards mentioned, most of which arose in the 70s, 80s and 90s.
An effort to standardize at this stage is completely natural, and like in all other cases mentioned, this will certainly bring huge benefits to users!

The Array API Standard
----------------------

The `Consortium for Python Data API Standards <https://data-apis.org>`_ are on a mission to create this shared standard.
At Ivy, we support their efforts 100% and we are in the process of adopting their standard ourselves.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/background/standardization/consortium.png?raw=true
   :align: center
   :width: 25%

The consortium is lead by `Quansight <https://www.quansight.com>`_, an open-source-first company made up of leading open-source developers from the following organizations: `PyTorch <https://pytorch.org>`_, `NumPy <https://numpy.org>`_, `Pandas <https://pandas.pydata.org>`_, `SciPy <https://scipy.org>`_, `conda <https://docs.conda.io/en/latest/>`_, `dask <https://dask.org>`_, `Apache <https://www.apache.org>`_, `MXNet <https://mxnet.apache.org/versions/1.9.0/>`_, `ONNX <https://onnx.ai>`_, `scikit-learn <https://scikit-learn.org/stable/>`_, `Jupyter <https://jupyter.org>`_, `AWS <https://aws.amazon.com/free/?trk=ps_a134p000003yhYiAAI&trkCampaign=acq_paid_search_brand&sc_channel=ps&sc_campaign=acquisition_DACH&sc_publisher=google&sc_category=core-main&sc_country=DACH&sc_geo=EMEA&sc_outcome=Acquisition&sc_detail=aws&sc_content=Brand_Core_aws_e&sc_matchtype=e&sc_segment=456911458944&sc_medium=ACQ-P|PS-GO|Brand|Desktop|SU|Core-Main|Core|DACH|EN|Text&s_kwcid=AL!4422!3!456911458944!e!!g!!aws&ef_id=Cj0KCQiA6NOPBhCPARIsAHAy2zCeKSJAfsJ5BSqbOt0QsZpGXRE4h2MW06eJ_VchjwcOPQTVTPZsFvIaAgCiEALw_wcB:G:s&s_kwcid=AL!4422!3!456911458944!e!!g!!aws&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all>`_, `CuPy <https://cupy.dev>`_, `RAPIDS <https://developer.nvidia.com/rapids>`_, `.NET <https://dotnet.microsoft.com/en-us/>`_, `SymPy <https://www.sympy.org/en/index.html>`_, `Ray <https://www.ray.io>`_, `modin <https://modin.readthedocs.io/en/stable/>`_ and `Spyder <https://www.spyder-ide.org>`_.
Other collaborators include members of: `TensorFlow <https://www.tensorflow.org>`_, `JAX <https://jax.readthedocs.io/en/latest/>`_, `Google <https://about.google>`_, `OctoML <https://octoml.ai>`_ and `einops <https://einops.rocks>`_.
Further, the consortium is sponsored by `LG Electronics <https://mail.google.com/chat/u/0/#chat/dm/lZAjU4AAAAE>`_, `Microsoft <https://www.microsoft.com/en-us>`_, `Quansight <https://www.quansight.com>`_, `D E Shaw and Co <https://www.deshaw.com>`_, `TensorFlow <https://www.tensorflow.org>`_, `JAX <https://jax.readthedocs.io/en/latest/>`_ and `Intel <https://www.intel.com/content/www/us/en/homepage.html>`_.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/background/standardization/array_api_backers.png?raw=true
   :align: center
   :width: 100%

Together, all major ML frameworks are involved in the the Array API standard in one way or another.
This is a promising sign in the pursuit of unification.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/logos/supported/frameworks.png?raw=true
   :align: center
   :width: 60%


Clearly a lot of time, thought and careful attention has gone into creating the `first version <https://data-apis.org/array-api/latest/>`_ of the standard, such that it simplifies compatibility as much as possible for all ML frameworks.

We are very excited to be working with them on this standard, and bringing Ivy into compliance, with the hope that in due time others also follow-suit!


**Round Up**

Hopefully this has given some clear motivation for why standardization in ML frameworks could be a great thing, and convinced you that we should celebrate and encourage the foundational work by the Array API Standard 🙂

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!
