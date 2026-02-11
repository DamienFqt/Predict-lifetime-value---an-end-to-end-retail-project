# To be completed

# About
This project serves two purposes:
- to showcase a complete end-to-end workflow for a typical retail problem, from the business problem to the model deployment and its monitoring;
- to strengthen practical skills in MLOps, best coding practices, and version control with Git.

It is based on a fictitious company - selling violette-based products (see below its logo) - and synthetic data whose process is clearly defined. 

<p align="center">
  <img width="128" height="128" alt="Logo Vuciliette" src="images\Logo Vuciliette.png"> [1]
</p>

The final goal is to answer the business problem by deploying an application which can predict the top customers according to their lifetime value.

--- 

# Business problem
Vuciliette aims to launch a fully personalized, long-term loyalty program designed to strengthen customer relationships and reward engagement over time. The program is intended to be highly inclusive, offering increasing value to customers as their relationship with the brand grows.
According to a study conducted by an external consulting firm, the initiative could generate up to €1,000 in additional annual value per long-term top customer over an average tenure of 10 years, for an estimated cost of €300 per customer.
Accurately identifying **high-value customers based on their lifetime value** is therefore a **key strategic challenge** for the effective rollout of this long-term loyalty program.

---

# Detailed Documentation

Detailed documentation, including model specification and potential improvements, can be found [here](Documentation/Detailed_documentation.md). It includes:

- The model specification
- The definition of model versioning
- A list of potential improvements
- Suggested monitoring metrics, including the costs and benefits of correctly identifying top customers

---

# Data Source

The project is based on synthetic data. The data generating process can be found [here](Documentation/Data_Generating_Process.md).

---

# Results : Early Solution Overview
An application has been deployed that allows business teams to choose two parameters:
- Number of clients
- Model

The application then displays top customers according to their predicted lifetime value. Metrics showing potential costs and gains are also provided.


# Getting started

## Local installation
1. Clone the repository
```bash
git clone https://github.com/DamienFqt/Predict-lifetime-value---an-end-to-end-retail-project.git
```
2. Create the environment (replace `<your_env>` by the name of your environment)
```bash
conda env create -f requirements.yml -n <your_env>
conda activate <your_env>
```


# Usage 
Connect to the interface allows you to modify :
- the number of top customers for whom you would like the lifetime value
- the model version

As a fictitious company's employee, you can then copy and paste the list of top customers and contact them.

## Author
Damien FOUQUET

## References
[1] Image générée par IA via ChatGPT (OpenAI GPT-5).  
Commande utilisée : "Créer un logo centré pour Vuciliette, 128x128 px, style minimaliste, avec une violette illustrée".  



