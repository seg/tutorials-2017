## FWI modeling, part 1: Forward modeling

The manuscript is in a runnable Jupyter Notebook — [manuscript.ipynb](notebooks/manuscript.ipynb). It contains instructions for setting up your environment and installing Devito.

See [Part 2](https://github.com/seg/tutorials-2018/blob/master/1801_FWI_Adjoint_modeling/notebook/Notebook.ipynb) and [Part 3](https://github.com/seg/tutorials-2018/blob/master/1802_FWI_Inversion/Notebook/Manuscript.ipynb).

### Install Devito

    git clone -b v3.1.0 https://github.com/opesci/devito
    cd devito
    conda env create -f environment.yml
    source activate devito
    pip install -e .

Thank you,
The Authors
