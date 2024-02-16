# UncertainDepths

## Estimating depths from a single image, with *reliable* uncertainty.


:car: Online and Posthoc Laplace gives *intuitive* uncertainties - models trained on roads and cars (KITTI), are confident in predictions on roads and cars. :car:


### Uncertainties on test-set images


![Uncertainties given by our models, and baselines](UncQualReport.png)



While other methods fail in predicting high uncertainty -> high loss, Online Laplace is reliable across all standard loss metrics.


### Only Online Laplace has monotonically increasing error in uncertainty.

While other methods fail in predicting high uncertainty -> high loss, Online Laplace is reliable across all standard loss metrics.

![Monotonicity of uncertainty w.r.t. common losses differ by model type](src/final_reportRoot_Mean_Squared_Error.png)



:hourglass: Online Laplace uses 70 % less training compute than ensembles for similar results. :hourglass:

Project Organization
------------

    ├── README.md         
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
