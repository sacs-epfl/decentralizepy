==============
decentralizepy
==============

-------------------------
Setting up decentralizepy
-------------------------

* Fork the repository.
* Clone and enter your local repository.
* Check if you have ``python>=3.8``.
* (Optional) Create and activate a virtual environment.
* Update pip. ::

    pip3 install --upgrade pip
    pip install --upgrade pip

* On Mac M1, installing ``pyzmq`` fails with `pip`. Use ``conda``.
* Install decentralizepy for development. ::

    pip3 install --editable .\[dev\]
    
----------------
Running the code
----------------

* Choose and modify one of the config files in ``eval/{step,epoch}_configs``.
* Modify the dataset paths and ``addresses_filepath`` in the config file.
* In eval/run.sh, modify ``first_machine`` (used to calculate machine_id of all machines), ``original_config``, and other arguments as required.
* Execute eval/run.sh on all the machines simultaneously. There is a synchronization barrier mechanism at the start so that all processes start training together.

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
