# Budworm Addiction Modeling Framework

This repository contains code to implement the modeling framework discussed in Boot et al. (2025, under review) and reproduce all simulations from the manuscript.
The modeling framework integrates a dual-process model of addiction in an agent-based network model.

---

## 🐛 Overview


- **Ecological dynamics**: Simulating spruce budworm population changes with predator-prey interactions.
- **Bifurcation analysis**: Visualizing multiple population steady states across predator strength.
- **Social mechanisms**: Modeling how social behavior influences ecological outcomes (e.g., bird predation), or simulating collective phenomena like protests or behavior adoption.
- **Behavioral transitions**: Simulating how people switch between smoking, vaping, dual use, or abstention.

---

## 🧭 Project Structure

```
Budworm-addiction/
├── models/                  # Core model implementations
│   ├── budworm_model.py             - Basic predator-prey dynamics
│   ├── bifurcation_triple.py        - Bifurcation analysis of predator strength
│   ├── sigmoid_social_B.py          - Social feedback on predation
│   ├── budworm .py                  - Main budworm model code
│   ├── N_over_t_with_sens.py        - ...
│   ├── social_phenomena_sim.py      - Threshold-based adoption model
│   └── vaping_and_smoking_sim.py    - Behavior switching (Markov-style)
├── notebooks/              # Example Jupyter notebooks
│   └── budworm_model_demo.ipynb
├── tests/                  # Basic test coverage
│   └── test_budworm_model.py
├── requirements.txt        # Python dependencies
├── setup.py                # Python package setup (optional)
└── README.md               # You are here
```

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Jesse291847/Budworm-modeling-framework.git
cd Budworm-addiction
pip install -r requirements.txt
```

---

## 🚀 Usage

Each script in `models/` can be run independently or imported into a notebook. For example:

```bash
python models/budworm_model.py
```

### Or use the demo notebook:

```bash
jupyter notebook notebooks/budworm_model_demo.ipynb
```

---

## 📘 Explanation per Script

### Main Scripts — Used to produce the Figures in the Main Text

These scripts reproduce the main figures in the manuscript (*Boot et al., 2025, under review*).

#### 🔹 `budworm_dynamics.R`
- Demonstrates the basic dynamics of the Spruce Budworm Outbreak Model.
- Reproduces **Figure 2**.

#### 🔹 `bifurcation_analysis.R`
- Performs bifurcation analysis, identifying stable and unstable states.
- Reproduces **Figure 3**.

#### 🔹 `example_stable_states.R` [STILL DO IN R]
- Illustrates convergence to different stable states across the bifurcation diagram.
- Reproduces **Figure 4**.

#### 🔹 `multiple_stable_states.R` [ADD RIGHT PANEL]
- Simulates stable patterns of use and transitions to heavy use.
- Reproduces **Figure 5**.

#### 🔹 `relapse_quitting.R` [STILL DO IN R]
- Models sudden quitting and relapse.
- Reproduces **Figure 6**.

#### 🔹 `sudden_epidemic.R`
- Simulates sudden outbreaks of use via social influence.
- Reproduces **Figure 7** and **Figure E1**.

#### 🔹 `disruption_social_network.R` [Organize sim_data folder]
- Simulates recovery via disruption of a harmful social environment.
- Reproduces **Figure 8** and **Figure F1**.

#### 🔹 `illegal_legal.R`
- Models clustering patterns among users of legal vs. illegal substances.
- Reproduces **Figure 9**.

#### 🔹 `vaping_smoking.R` [MAKE NEAT]
- Extends the model to competitive/mutualistic substance relationships.
- Reproduces **Figure 10**.

#### 🔹 `social_hysteresis.R` [MAKE NEAT]
- Demonstrates social hysteresis and irreversibility.
- Reproduces **Figure 11**.

---

### Helper Scripts — Called by Other Scripts

These are utility or support scripts used internally by the main ones, but do not generate figures directly.

*(Add scripts here as you identify them)*

---

### 📎 Supplementary Material Scripts

These scripts reproduce figures in the Supplementary Materials of the manuscript.

#### 🔹 `...` *(Add any relevant ones here once listed)*



## 🧪 Testing

Run the included unit tests using `pytest`:

```bash
pytest tests/
```

---

## 📈 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `numpy`
- `matplotlib`
- `scipy`
- 'pandas'
- 'networkx'
- 'ipywidgets'


---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork this repository.
2. Create a new branch (`git checkout -b feature-xyz`)
3. Commit your changes (`git commit -am 'Add feature xyz'`)
4. Push to the branch (`git push origin feature-xyz`)
5. Create a Pull Request

---

## 📜 License

MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgments

Originally built from a mixed R/Python framework, this package now follows Pythonic conventions for scientific modeling and simulation.
