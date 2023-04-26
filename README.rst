.. image:: https://upload.wikimedia.org/wikipedia/commons/f/f4/Logo_EPFL.svg
   :alt: EPFL logo
   :width: 75px
   :align: right

==============
decentralizepy
==============

decentralizepy is a framework for running distributed applications (particularly ML) on top of arbitrary topologies (decentralized, federated, parameter server).
It was primarily conceived for assessing scientific ideas on several aspects of distributed learning (communication efficiency, privacy, data heterogeneity etc.).

-------------------------
Setting up decentralizepy
-------------------------

* Fork the repository.
* Clone and enter your local repository.
* Check if you have ``python>=3.8``. ::

    python --version

* (Optional) Create and activate a virtual environment. ::
  
    python3 -m venv [venv-name]
    source [venv-name]/bin/activate

* Update pip. ::

    pip3 install --upgrade pip
    pip install --upgrade pip

* On Mac M1, installing ``pyzmq`` fails with `pip`. Use `conda <https://conda.io>`_.
* Install decentralizepy for development. (zsh) ::

    pip3 install --editable .\[dev\]
    
* Install decentralizepy for development. (bash) ::

    pip3 install --editable .[dev]
 
----------------
Running the code
----------------

* Choose and modify one of the config files in ``eval/{step,epoch}_configs``.
* Modify the dataset paths and ``addresses_filepath`` in the config file.
* In eval/run.sh, modify arguments as required.
* Execute eval/run.sh on all the machines simultaneously. There is a synchronization barrier mechanism at the start so that all processes start training together.

------
Citing
------

Cite us as ::

    @article{dhasade2023decentralized,
    title={Decentralized Learning Made Easy with DecentralizePy},
    author={Dhasade, Akash and Kermarrec, Anne-Marie and Pires, Rafael and Sharma, Rishi and Vujasinovic, Milos},
    journal={arXiv preprint arXiv:2304.08322},
    year={2023}
    }

------------
Contributing
------------

* ``isort`` and ``black`` are installed along with the package for code linting.
* While in the root directory of the repository, before committing the changes, please run ::

    black .
    isort .

-------
Modules
-------

Following are the modules of decentralizepy:

Node
----
* The Manager. Optimizations at process level.

Dataset
-------
* Static

Training
--------
* Heterogeneity. How much do I want to work?

Graph
-----
* Static. Who are my neighbours? Topologies.

Mapping
-------
* Naming. The globally unique ids of the ``processes <-> machine_id, local_rank``

Sharing
-------
* Leverage Redundancy. Privacy. Optimizations in model and data sharing.

Communication
-------------
* IPC/Network level. Compression. Privacy. Reliability

Model
-----
* Learning Model
