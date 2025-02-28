# SATELLITE

## VENV

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Where `requirements.txt` is:

```txt
MarkupSafe 3.0.2,
PyYAML 6.0.2,
Send2Trash 1.8.3,
anyio 4.8.0,
argon2-cffi 23.1.0,
argon2-cffi-bindings 21.2.0,
arrow 1.3.0,
asttokens 3.0.0,
async-lru 2.0.4,
attrs 25.1.0,
babel 2.17.0,
beautifulsoup4 4.13.3,
bleach 6.2.0,
certifi 2025.1.31,
cffi 1.17.1,
cftime 1.6.4.post1,
charset-normalizer 3.4.1,
comm 0.2.2,
contourpy 1.3.1,
cycler 0.12.1,
debugpy 1.8.12,
decorator 5.1.1,
defusedxml 0.7.1,
executing 2.2.0,
fastjsonschema 2.21.1,
fonttools 4.56.0,
fqdn 1.5.1,
h11 0.14.0,
httpcore 1.0.7,
httpx 0.28.1,
idna 3.10,
ipykernel 6.29.5,
ipython 8.32.0,
ipywidgets 8.1.5,
isoduration 20.11.0,
jedi 0.19.2,
jinja2 3.1.5,
joblib 1.4.2,
json5 0.10.0,
jsonpointer 3.0.0,
jsonschema 4.23.0,
jsonschema-specifications 2024.10.1,
jupyter 1.1.1,
jupyter-client 8.6.3,
jupyter-console 6.6.3,
jupyter-core 5.7.2,
jupyter-events 0.12.0,
jupyter-lsp 2.2.5,
jupyter-server 2.15.0,
jupyter-server-terminals 0.5.3,
jupyterlab 4.3.5,
jupyterlab-pygments 0.3.0,
jupyterlab-server 2.27.3,
jupyterlab-widgets 3.0.13,
kiwisolver 1.4.8,
matplotlib 3.10.0,
matplotlib-inline 0.1.7,
mistune 3.1.2,
nbclient 0.10.2,
nbconvert 7.16.6,
nbformat 5.10.4,
nest-asyncio 1.6.0,
netCDF4 1.7.2,
notebook 7.3.2,
notebook-shim 0.2.4,
numpy 2.2.3,
overrides 7.7.0,
packaging 24.2,
pandocfilters 1.5.1,
parso 0.8.4,
pexpect 4.9.0,
pillow 11.1.0,
pip 25.0.1,
platformdirs 4.3.6,
prometheus-client 0.21.1,
prompt-toolkit 3.0.50,
psutil 7.0.0,
ptyprocess 0.7.0,
pure-eval 0.2.3,
pycparser 2.22,
pygments 2.19.1,
pyparsing 3.2.1,
python-dateutil 2.9.0.post0,
python-json-logger 3.2.1,
pyzmq 26.2.1,
referencing 0.36.2,
requests 2.32.3,
rfc3339-validator 0.1.4,
rfc3986-validator 0.1.1,
rpds-py 0.23.0,
scikit-learn 1.6.1,
scipy 1.15.2,
setuptools 75.8.0,
six 1.17.0,
sniffio 1.3.1,
soupsieve 2.6,
stack-data 0.6.3,
terminado 0.18.1,
threadpoolctl 3.5.0,
tinycss2 1.4.0,
tornado 6.4.2,
traitlets 5.14.3,
types-python-dateutil 2.9.0.20241206,
typing-extensions 4.12.2,
uri-template 1.3.0,
urllib3 2.3.0,
wcwidth 0.2.13,
webcolors 24.11.1,
webencodings 0.5.1,
websocket-client 1.8.0,
widgetsnbextension 4.0.13,
autocommand 2.2.2,
backports.tarfile 1.2.0,
importlib-metadata 8.0.0,
inflect 7.3.1,
jaraco.collections 5.1.0,
jaraco.context 5.3.0,
jaraco.functools 4.0.1,
jaraco.text 3.12.1,
more-itertools 10.3.0,
tomli 2.0.1,
typeguard 4.3.0,
wheel 0.43.0,
zipp 3.19.2
```

## DOCKER

### Build

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/pytorch-env:latest --push .
```

```bash
docker pull yourusername/pytorch-env:latest
```


### Run

```bash
docker run -it --gpus all -v /path/to/local/repo:/workspace/repo -v /path/to/local/data:/data yourusername/pytorch-env:latest
```

Then from the container go to the dataset directory, in which you will find the scripts for preprocessing the data.

```bash
cd /workspace/repo/dataset
```

Then from there run `python` and follow the steps in the script.

```python
from utils import process_directory_tree
input_dir = '/data/GridRad/'
output_dir = '/data/Output/'
process_directory_tree(input_dir=input_dir, output_dir=output_dir, checkpoint_file='checkpoint_file', n_workers=10)
```


## APPTAINER

### Build

To run the code on ALCF Polaris supercomputer, we will use the Apptainer containerization tool.

For more information on how to run jobs on Polaris, please refer to the following link:
[Polaris](https://docs.alcf.anl.gov/polaris/running-jobs/)

For more information on how to manage containers on Polaris, please refer to the following link:
[Apptainer](https://docs.alcf.anl.gov/polaris/containers/containers/)


Once you have access to polaris, you can use the following commands to run the code on the supercomputer.

First, you need to login to the compute node.
Use the following command  to login interactively to the compute node:

```bash
qsub -I -l select=1 -l filesystems=home:eagle:grand -l walltime=1:00:00 -q debug -A <Projectname>
```

Then, you need to load the required modules:

```bash
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
ml use /soft/modulefiles
ml load spack-pe-base/0.8.1
ml load apptainer
ml load e2fsprogs
```

For exporting the Docker image to Apptainer, use the following command:

```bash
apptainer build satellite.sif docker://yourusername/satellite:latest
```

Please, take into account that the `satellite.sif` file will be created in the current directory where you run the command.
It is important to have enough space in the directory where you run the command to store the image.
Therefore, pay attention to your disk quota.

In polaris you can use `myquota` or `myprojectquotas` to check your disk quota.
It is recommended to use you project directory to store the image.

It is important to build the image in the compute node, where all the cuda libraries and GPU drivers are available.

After the image has been created, you can run the code using the following command:

```bash
apptainer shell -B /path/to/your/data:/data satellite.sif
```

Once inside Apptainer, go to the directory `dataset` inside the repo:

```bash
Apptainer> cd ~/SATELLITE/dataset/
```

Then run the preprocessing script:

```bash
Apptainer> python
Python 3.12.3 (main, Nov  6 2024, 18:32:19) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
Could not open PYTHONSTARTUP
FileNotFoundError: [Errno 2] No such file or directory: '/etc/pythonstart'
>>> from utils import process_directory_tree
>>> input_dir = '/data/SATELLITE/asdc.larc.nasa.gov'
>>> output_dir = '/data/SATELLITE/Output'
>>> process_directory_tree(input_dir=input_dir, output_dir=output_dir, checkpoint_file='checkpoint_file', n_workers=100)
```

This script will preprocess the data and save the output in the `output_dir` directory.

In the following figures we can see the results of the preprocessing step.

<img src="figures/Preprocessing1.png" alt="Pre-processing result" width="800"/>

<img src="figures/Preprocessing2.png" alt="Pre-processing result" width="800"/>
