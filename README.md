## Conditional Neural Processes

This repository implements Conditional Neural Processes by Garnelo et al. 2018.

### Getting Started in Google Colab
1. In order to get started simply clone this repository.
``git clone https://github.com/fredchettouh/neural-processes.git``
2. Add this directory to your google drive 
3. Go to the ``experiment`` directory and open the ``1d_gp_compare_aggregation.ipynb`` notebook. Follow the instructions in the notebook 
to mount your google drive 
4. Run the experiment 
### Running an experiment Locally 
1. In order to get started simply clone this repository.
``git clone https://github.com/fredchettouh/neural-processes.git``
2. Run ``pip3 install -r requirements.txt``
3. Go to the ``experiment`` directory and open the ``1d_gp_compare_aggregation.ipynb`` notebook.
4. Run the experiment 

### The configuration files:
This library has a simple API in form of configuration files. In the base configurations you can find base configs for the three cases that are currently implemented as examples a) 1D Few Shot Regression with GP generated data b) Multivariate Few Shot Polynomial Regression c) 2D Image reconstruction with Pixel with pixel regression.
Open up a new notebook using the poilerplate in the ``experiment`` directory, load a base configuration file and change it according to your needs. 

### Implementing new use cases

To apply CNPs to new use cases the first step is to think about the data generation step. The ``Datagenerator`` in ``cnp/datageneration.py`` can be extended by any processes to produce data or sample from a data set. Simply ensure that it implements the ``generate_curves`` method which has to return tensors with x,y values that can be turned to data loaders. 
Once this is done simply indicate that this method should be loaded dynamically in the configuration file for your experiment.
