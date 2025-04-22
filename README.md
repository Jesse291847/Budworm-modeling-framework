# Budworm Addiction Modeling Framework

This repository contains code to implement the modeling framework discussed in Boot et al. (2025, under review) and reproduce all simulations from the manuscript.
The modeling framework integrates a dual-process model of addiction in an agent-based network model. 

---

## 📘 Explanation per Script [Missing scripts will be added in the week of 21st of april]

### Main Scripts — Used to produce the Figures in the Main Text

These scripts reproduce the main figures in the manuscript (*Boot et al., 2025, under review*).

#### 🔹 `budworm_dynamics.R`
- Demonstrates the basic dynamics of the Spruce Budworm Outbreak Model.
- Reproduces **Figure 2**.

#### 🔹 `bifurcation_analysis.R`
- Performs bifurcation analysis, identifying stable and unstable states.
- Reproduces **Figure 3**.

#### 🔹 `example_stable_states.R`
- Illustrates convergence to different stable states across the bifurcation diagram.
- Reproduces **Figure 4**.

#### 🔹 `multiple_stable_states.R`
- Simulates stable patterns of use and transitions to heavy use.
- Reproduces **Figure 5**.

#### 🔹 `relapse_quitting.R`
- Models sudden quitting and relapse.
- Reproduces **Figure 6**.

#### 🔹 `sudden_epidemic.R`
- Simulates sudden outbreaks of use via social influence.
- Reproduces **Figure 7** and **Figure E1**.

#### 🔹 `disruption_social_network.R`
- Simulates recovery via disruption of a harmful social environment.
- Reproduces **Figure 8** and **Figure F1**.

#### 🔹 `illegal_legal.R`
- Models clustering patterns among users of legal vs. illegal substances.
- Reproduces **Figure 9**.

#### 🔹 `vaping_smoking.R`
- Extends the model to competitive/mutualistic substance relationships.
- Reproduces **Figure 10**.

#### 🔹 `social_hysteresis.R` 
- Demonstrates social hysteresis and irreversibility.
- Reproduces **Figure 11**.

---

### Helper Scripts — Called by Other Scripts
#### 🔹 `helper_functions.R`
- Contains the function to update the newtork structure
- Contains the funciton to color the nodes in the network according to consumption status

#### 🔹 `social_phenomena_setup.R`
- Sets up our a social network model not specific to any phenomenon or scenario
- Called in all social phenomena scrips
---
### Supplementary Material Scripts
These scripts reproduce figures and analyses in the appendicies of the manuscript.

### Tutorial scripts — 
These scripts contain miminimal running examples of our model not used in the manuscript.

#### 🔹 `individual_tutorial.R`
- Contains a tutorial on how to implement our model for one individual.

#### 🔹 `social_tutorial.R`
- Contains a minimal example that show how to implement our full modeling framework.
---

## 📜 License

MIT License. See the [LICENSE](LICENSE) file for details.

---

