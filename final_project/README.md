# Intro to Robotics: Final Project

## Overview

For the final project in this class, we will use imitation learning for vision-based drone navigation. The project will give you an opportunity to see machine learning for robotics in action, obtain a deeper understanding of imitation learning methods, and explore the impact of various design choices in terms of data, models, loss functions, and algorithms. 


### Main components:
1. Collect training data (See [README_IL](drone/datasets/utils/README_IL.md)).
2. Train your model.
3. Run testing.

Detailed instructions for each step are provided below. In addition, we have put together some basic tips for working with the Crazyflie [here](https://github.com/Princeton-Introduction-to-Robotics/F2025/blob/main/crazyflie-tips.md).

## Lab setup and grading

We will hold a Demo Day for evaluating final projects. This will be held on Dean's Date. At the beginning of December, we will send out a sign-up sheet for the Demo Day. Each team will sign up for a time-slot (20 minutes) and will have four trials through the obstacle course(s) setup in the lab. Each team will also explain the technical approach that they took to the course staff. 

Of the four trials, two will be in a "nominal" setup and two will test the generalization of your learned policy:

* Nominal (2 flights): we will set up 5 obstacles in a netted drone cage in the lab. These will be the same (blue) obstacles that are in the lab. On Demo Day, these obstacles will be placed in a new configuration, and your drone will navigate through the course.
* Generalization (lighting; 1 flight): same as nominal, but with half the lights off in the lab (there are two light switches in the lab; switching off one turns off 50% of the lights).
* Generalization (obstcle color; 1 flight): we will again set up 5 obstacles, but with randomly colored tape on them. The tape will be solid colored (i.e., no weird textures or patterns) wrapped around the obstacles. We will place some examples of taped obstacles in the lab for you to test on (but on Demo Day, we will use an unseen tape color).

Your score on each of the trials will be based on the following criterion: distance along the x (i.e., forward) direction your robot traversed before colliding. In particular, the score for a trial will be the fraction of the course your robot successfully traversed before colliding, e.g., 70/100 pts if your robot covered 70 percent of the course before colliding ("colliding" is defined as the point at which your robot first touches/hits an obstacle or the ground, or the netting).

Your total score will be the **average of the three best scores from the four trials**. The rationale for this is to evaluate the reliability of your system. Thus, robustness is the main feature to strive for in this project (as it has been throughout the course!).

## Generative AI and coding agents use

As a reminder, you are **allowed** to use generative AI and coding agents however you choose. These can be extremely helpful if you want to try out different network architectures (e.g., ConvNet vs. transformer), loss function choices (e.g., classificaion vs. regression), or data augmentation schemes (e.g., color jittering, random cropping, etc.). This is your opportunity to learn how to use generative AI well! 

## Installation

### Option 1: `mae345`
If you want to keep using the `conda` environment, install the following additional packages:

```bash
# Activate conda environment
conda activate mae345

# Install computer vision libraries
pip install opencv-python

# Install keyboard control
pip install pynput

# Install PyTorch (for data loading and training)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# OR for GPU:
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Option 2: UV
If you want to start with a fresh environment, we recommend using `uv`.
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).\
    On mac, run `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Add paths \
    ```bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```
3. Initialize environment \
    `uv sync`
4. Install custom package \
   `uv pip install -e .`
5. Activate environment
   - In terminal, run `source .venv/bin/activate`
   - In notebook, select interpreter to be `.venv/bin/python`




## Repo Structure

### Configs
To streamline drone inference and testing, we use `hydra`. This package supports easily swapping
configuration sets (e.g., model selection, runtime options) without having to change code. 

Configs are stored under `drone/configs/`. You can make your own config file and keep track of all the arguments you used in a particular experiment!

### Main Entrypoints
- `drone/scripts/training_model.ipynb`: This is the colab script you will use for training the action prediction model. You should upload this to [Google Colab](https://colab.research.google.com/) to run the code. See detailed instructions in the file.
- `drone/scripts/fly.py`: This is the main testing script that flies the drone in closed-loop. 
  - We provide an example config file in `drone/configs/example_fly_config.yaml`. 
  - Feel free to change the existing parameters, and add any other parameters you want!


### Datasets
See instructions in
[`README_IL`](drone/datasets/utils/README_IL.md) about:
- Data collection
- Data analysis
- PyTorch DataLoader

We provide one example trajectory under `drone/datasets/imitation_data`. You can download an example dataset with 15 trajectories [here](https://drive.google.com/file/d/1YLRJ-F8YpfLPHny5antS8u4yIrfjPOz3/view?usp=sharing).

### Models
We provide two example models:
- `continuous_action_model`,
- `discrete_action_model`.
  
Please refer to the repective files for more details.

**Model Configs:**
Model-specific options (architecture, pretrained flag, action dimension, etc.) live under
`drone/configs/models/`. These config files are meant to be selected by Hydra at runtime so you
can switch between model implementations or hyperparameters without changing code. Feel free to adjust these parameters as you see fit!

In the final project, feel free to explore and add your own model class, following these steps:
1. Create a new python file under `drone/models/`.
2. Create a superclass of `nn.Module` following the format of the examples.
3. Make sure to include a `output_to_executable_actions` function to map the model output to actions we actually command the drone with!
4. Add a configuration file under `drone/configs/models/` following other examples.

### Controls
Basic drone control class that supports 
- hovering
- sending specific motion commands
- taking an image

Feel free to add more functionalities!

### Scripts
These are the main entrypoints for data collection, model training, and closed-loop testing!

