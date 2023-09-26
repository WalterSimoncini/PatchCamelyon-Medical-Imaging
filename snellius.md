# Running experiments on Snellius

This guide assumes that you've received a Snellius account and you've set up ssh access, so that you can open a ssh session by running

```sh
ssh user@snellius.surf.nl  
```

## Setup

Before you start, you want to load the 2022 and anaconda modules, to do so run:

```sh
module load 2022
module load Anaconda3/2022.05
```

Create a conda environment and activate it with the following commands

```sh
conda create -n medical-imaging python=3.10
source activate medical-imaging
```

Compress the repository and copy it to your home folder with the following command:

```sh
tar -czf pcam.tar.gz pcam
scp pcam.tar.gz user@snellius.surf.nl:/home/user
```

Unpack the archive on Snellius with

```sh
tar -xf pcam.tar.gz
```

Enter the folder and install the required packages

```sh
cd pcam
pip install -r requirements.txt
```

Dowload the data by running

```sh
sh download.sh
```

And finally log into wandb to track jos:

```sh
wandb login
```

## Running Jobs

Some example jobs are stored in the `jobs` folder and you can use them as a template. For more information about job files refer to [this article](https://servicedesk.surf.nl/wiki/display/WIKI/Example+job+scripts) and for some general information [read this one](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html).

You can schedule a job with

```sh
sbatch path/to/job.job
```

This command will print out the job id. To check its status run:

```sh
scontrol show job <job_id>
```

If you need to cancel a job run:

```sh
scancel <job_id>
```
