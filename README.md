# Risky Ageing

The code in this repository produces the results in S. W. Omholt, T. B. L. Kirkwood, **Aging as a consequence of selection to reduce the environmental risk of dying**, *Proc. Natl. Acad. Sci. U.S.A.* **?**, ??-??, 2021.

The paper presents three case studies that are used to validate the evolutionary relevance of the main hypothesis underlying the paper. Case study 1 focuses on the bowl and doily spider (*Frontinella pyramitela*) (Figures 1, 2 and 3). Case study 2 focuses on the dipterian *Telostylinus angusticollis* (Figure 4). Case study 3 (Figure 5) is somewhat more loosely linked to experimental data, but is closely guided by life-history data of the European lobster (*Homarus gammarus*). 

The main intention with this repository is to allow others to check that the code faithfully reflects and reproduces what is written and presented in the paper. The case studies in this repository are complete only when read together with their corresponding text in the paper.

It is our hope that the underlying code can be used in teaching as well in research. Concerning the latter, if you possess survivorship data for an organism in the wild as well as in a protected environment, it should be straight-forward to use the code to test how well our life-history model can accommodate for your data.

The original code has most kindly been fully refactored by [Oda Omholt](https://github.com/odaom) in order to ease readability and reuse.

## Abstract

Each animal in the Darwinian theatre is exposed to a number of abiotic and biotic risk factors causing mortality. Several of these risk factors are intimately associated with the act of energy acquisition as such, and with the amount of reserve the organism has available from this acquisition for overcoming temporary distress. Because a considerable fraction of an individual’s lifetime energy acquisition is spent on somatic maintenance, there is a close link between energy expenditure on somatic maintenance and mortality risk. Here we show, by simple life-history theory reasoning backed up by empirical cohort survivorship data, how reduction of mortality risk might be achieved by restraining allocation to somatic maintenance, which enhances lifetime fitness but results in aging. Our results predict the ubiquitous presence of senescent individuals, in a highly diverse group of natural animal populations, which may display constant, increasing or decreasing mortality with age. This suggests that allocation to somatic maintenance is primarily tuned to expected lifespan by stabilizing selection and is not necessarily traded against reproductive effort or other traits. Due to this ubiquitous strategy of modulating the somatic maintenance budget so as to increase fitness under natural conditions, it follows that individuals kept in protected environments with very low environmental mortality risk, will have their expected lifespan primarily defined by somatic damage accumulation mechanisms laid down by natural selection in the wild.

## Installation
With Python version >=3.8, install the required packages.

```bash
pip install -r requirements.txt
```

## Usage
To inspect and run the case studies, open the notebooks in JupyterLab.

```bash
jupyter lab
```

The case studies are pre-run for immediate inspection, and may be further experimented with for a closer look. The `%%time` cell magic outputs show the approximate times one can expect for reruns of the most time-consuming simulations.

The logic of the cohort model and fitness calculations can be found in the modules `cohort_model` and `fitness`, respectively. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)