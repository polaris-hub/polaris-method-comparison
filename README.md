# Polaris - Method Comparison Guidelines
[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2024--6dbwv--v2-blue)](https://doi.org/10.26434/chemrxiv-2024-6dbwv-v2)

This repository includes the code for the [Practically significant method comparison protocols for machine learning in small molecule drug discovery](https://doi.org/10.26434/chemrxiv-2024-6dbwv-v2) preprint. This work is part of the [Polaris](https://polarishub.io/guidelines/small-molecules) initiative. To learn more, see also our [Nature Machine Intelligence Correspondence](https://doi.org/10.1038/s42256-024-00911-w).

## Webinar
We hosted [a webinar](https://github.com/polaris-hub/polaris-method-comparison/discussions/6) on December 5th to present the paper. 

You can find the recording here: https://www.youtube.com/watch?v=qaqw2wNNdqE

## We would love to hear from you!
We've done our best to come up with sensible guidelines, but would love to hear from you. Is there anything we missed? The best way to get in touch is by starting a Github discussion in this repository.

We're also working with the team at www.polarishub.io to design a novel way of evaluation and comparing methods in drug discovery that goes beyond the typical leaderboard. If you're interested in helping us shape these ideas by giving feedback on early designs, [please reach out](https://github.com/polaris-hub/polaris-method-comparison/discussions).

## Where to go from here?
To simplify adoption of the proposed guidelines, this repository includes some code snippets and examples that can hopefully help.

## Important Note
To use the software in this repo, you must first install GitHub Large File Storage (LFS). For more information on LFS, please see this page.

https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage


To install dependencies:

```
pip install -r requirements.txt
```

The primary statistical testing workflow we are recommending is here: `ADME_example/ML_Regression_Comparison.ipynb`.

Additional code of interest:

1. **Case study:** All code related to the case study discussed in Section 3.3.1 can be found in the the `ADME_example/` folder.
2. **Experiment (CV):** All code related to the experiment discussed in Appendix B can be found in the `repeated_cv_simulation/` folder.
3. **Figure (Dynamic Range)**: All code related to Figure 4 can be found in the `Dynamic_Range_example/` folder.

## How to cite
[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2024--6dbwv--v2-blue)](https://doi.org/10.26434/chemrxiv-2024-6dbwv-v2)

```
Ash JR, Wognum C, Rodríguez-Pérez R, Aldeghi M, Cheng AC, Clevert D-A, et al.
Practically significant method comparison protocols for machine learning in small molecule drug discovery.
ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-6dbwv-v2
This content is a preprint and has not been peer-reviewed.
```

```bib
@article{ash2024practically,
  title={Practically significant method comparison protocols for machine learning in small molecule drug discovery.},
  author={Ash, Jeremy R and Wognum, Cas and Rodr{\'\i}guez-P{\'e}rez, Raquel and Aldeghi, Matteo and Cheng, Alan C and Clevert, Djork-Arn{\'e} and Engkvist, Ola and Fang, Cheng and Price, Daniel J and Hughes-Oliver, Jacqueline M and others},
  year={2024}
}
```
