# UncertainDepths

## Estimating depths from a single image, with *reliable* uncertainty.


:car: Online and Posthoc Laplace gives *intuitive* uncertainties - models trained on roads and cars (KITTI), are confident in predictions on roads and cars. :car:


### Uncertainties on test-set images


![Uncertainties given by our models, and baselines](UncQualReport.png)





### Only Online Laplace has monotonically increasing error in uncertainty.

While other methods fail in predicting high uncertainty -> high loss, Online Laplace is reliable across all standard loss metrics.

![Monotonicity of uncertainty w.r.t. common losses differ by model type](src/final_reportRoot_Mean_Squared_Error.png)



:hourglass: Online Laplace uses 70 % less training compute than ensembles for similar results. :hourglass:

Project Organization
------------

    ├── README.md         
    │
    ├── report            <- Contains a report on the model, as well as some details and results on the runs
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── conf           <- Configs for running experiments, using hydra.
    |   |   
    │   ├── data           <- Scripts to download or generate data
    │   │   ├──── datamodules  <- datamodules for setting up dataloaders
    │   │   ├──── datasets <- scripts for setting up datasets
    │   │   └──── other scripts <- typically short and self-explanatory.
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├──── laplace/ <- implementation of class used for online laplace on top of lightning module
    │   │   ├──── modelImplementations/ <- implementations of the nnj-net used in report
    │   │   ├──── lightning_modules/ <-  lightning module supplying main training logic.
    │   │   ├──── outputs/ <-  outputs from all training runs - images, dataframes with uncertainty scores, weights, setup config.
    |   │   ├──── train_model.py <- trains the models according to config
    │   │   ├──── evaluate_model.py <- evaluates models - typically run by run_eval.py
    │   │   ├──── run_eval.py <- runs evaluate models
    │   │   ├──── viser_visualization_of_examples.py <- visualizes model prediction and uncertainty in 3D  
    │   │   └──── other scripts <- short and self explanatory
    │   │
    │   ├── paper_utility <- Small scripts for reproducing figures and tables in report.
    │   │
    │   ├── utility <- Miscellaneous scripts. Functions are small, few and self-explanatory
    │  
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## Reproducing

Experiments were run on a Nvidia A100 GPU, on a slurm cluster.

To reproduce the results from the report - run the respectively no_debug_run.sh by commenting out the relevant run. Then take the created folder in outputs and use it as input to the run_eval.sh bash script. 


