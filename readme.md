# Reliability-Adjusted Prioritized Experience Replay (ReaPER)

**ReaPER** is a novel transition selection scheme for online reinforcement learning technique that builds upon Prioritized Experience Replay (PER) by incorporating a reliability-adjusted temporal-difference error (TDE), which prioritizes experiences not just by surprise, but also by trustworthiness, enhancing learning efficiency.

This repository is an implementation of the Paper "Reliabiliy-adjusted Prioritized Experience Replay" (https://arxiv.org/abs/2506.18482, ICLR, 2026).


This repository builds upon the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) framework and includes an implementation of ReaPER integrated with a Double Deep Q-Network (DDQN) agent. It is intended for experimental use, prioritizes clarity over performance and is not optimized for runtime.

---

## Project Structure

```
.
├── example.py                          # Script showcasing example usage
├── requirements.txt                    # Required Python packages
├── README.md                           # Project overview and setup guide
├── custom_buffers/                    
│   └── reaper.py                       # ReaPER buffer implementation
└── custom_policies/
    └── custom_ddqn.py                  # DDQN agent modified for ReaPER
```

---

## Setup

To set up the environment:

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

To start training an agent, run:

```bash
python example.py
```

You can modify the training environment name directly in `example.py`.

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.  
See the [LICENSE](https://creativecommons.org/licenses/by/4.0/) for details.