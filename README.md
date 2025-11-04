# scXDR: A New Single-Cell Drug Response Prediction Model

**scXDR** is a novel drug response prediction model designed to operate across single-cell datasets. This model leverages advanced methodologies to predict how individual cells in tumor tissues respond to various drugs, providing new insights into personalized medicine and precision oncology.

## Key Features

- **Single-Cell Precision**: scXDR is specifically designed for single-cell data, addressing the complexities and heterogeneity within individual cells in tumor microenvironments.
  
- **Cross-Dataset Prediction**: The model can be applied across different single-cell datasets, enhancing its versatility and applicability in various research scenarios.

- **End-to-End & User-Friendly**: scXDR is an end-to-end, well-encapsulated framework with simple and efficient function calls, allowing users to smoothly perform drug response prediction.

## Applications

- **Cellular Response:**: scXDR enables the analysis of cellular responses to drugs at the single-cell level, revealing the heterogeneity and dynamic behavior of tumor cells.
  
- **Drug Screening**: The model can be used to screen and evaluate potential therapeutic compounds based on predicted single-cell drug responses.

- **Pan-Cancer Research**: scXDR supports cross-cancer studies based on pan-cancer single-cell datasets, contributing to a broader understanding of drug response mechanisms across multiple cancer types.

## Usage

- The example for running the code is in the `Example` folder, including the dataset download link and sample code usage.

- The environment information is in the `Environment` folder, including package versions and server configurations.

- The model scripts are in the `scXDR` folder, where `start.py` is used for the final execution.

**Please note:**  
This sample dataset is the same one used for the figures shown in the main text of the paper.  
You can obtain:  
- The **drug response value** for each cell  
- As well as directly derive the following metrics: **AUC**, **AUPR**, **Accuracy**, **F1 Score**, and **runtime**

---

**scXDR** represents a significant advancement in drug response prediction at the single-cell level, and we are excited about its potential to drive more effective cancer treatments and improve outcomes in precision medicine.
