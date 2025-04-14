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

## 📘 Model Descriptions

### 🔹 `budworm_model.py`
- Core logistic population growth with predation.
- Based on classic spruce budworm dynamics.

### 🔹 `bifurcation_triple.py`
- Sweeps over predator strength `B` to produce a bifurcation diagram.
- Visualizes multiple equilibria and sudden jumps in population.

### 🔹 `sigmoid_social_B.py`
- Introduces social feedback: collective action influences predator effectiveness.
- Predator strength `B` becomes a sigmoid function of budworm population.

### 🔹 `social_phenomena_sim.py`
- A threshold model where behavior (e.g., protesting, belief adoption) spreads once a tipping point is reached.

### 🔹 `vaping_and_smoking_sim.py`
- A behavior switching model: users can transition between smoking, vaping, both, or neither.
- Inspired by public health behavior models.

### 🔹 `budworm.py`
- 
- 

### 🔹 `N_over_t_with_sens.py`
- 
- 
---

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
