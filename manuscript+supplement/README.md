# Reproducing results of the `sbijax` manuscript

In order to reproduce the results of the paper, please follow the steps below.

Install a new Python3.11 virtual environment and activate it using:

```shell
python3.11 -m venv venv
source venv/bin/activate
```

Install `sbijax` and all dependencies using:
```shell
pip install -r requirements.txt
```

**NOTE**: In addiition, to be able to evaluate the notebook `chp5-real_data_example.ipynb` you need a working
`R`-installation. For our experiments, we used version `4.4.2 - Pile of Leaves`.

## Usage with Jupyter notebooks

For each section that contains code and figures, we provide a separate Jupyter notebook.
All notebooks contain the results and figures shown in the main manuscript.

 To run all experiments using Jupyter, first install a Jupyter kernel:
```shell
python -m ipykernel install --name sbi-dev --user
```

Then call
```shell
cd experimental_code
jupyter lab
```

This opens Jupyter on your web browser. You can now run any of the four notebooks.

## Usage with Python files

For each section that contains code and figures, we also provide separate Python files which
are exported from the notebooks.

To execute each and reproduce the results of a section, call:
```shell
cd experimental_code/scripts
python chp3-the_sbi_package.py
python chp4-examples.py
python chp5-real_data_example.py
python chpx-appendix.py
```

Calling a Python script creates files in a folder called `fig`.
