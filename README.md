# DS3ML3_Final_Project

Created by: Kyle Sung

Hi there! This is the main repository for my Final Project for Datasci 3ML3 in Winter 2024. Our goal was to create a Chess AI using machine learning techniques developed in the course.

**First**: If you haven't read the paper, please do this first. It can be found in the submission box.

**Next**: If you haven't tried playing agianst the bot, please do this next! 

1. Create an account on [lichess.org](https://lichess.org/).
2. In the top left corner, click **PLAY**. On the right side, click **PLAY WITH A FRIEND**.
3. Keep variant as **Standard**. Choose a **Real-Time** game with a finite length in minutes and seconds (the bot is unable to accept challenges of unlimited length). Choose your starting side!
4. In the section titled **Or invite a Lichess user**, enter `KS_ChessAI`
5. The game should begin. Enjoy!

#### Reproducability

This next section follows **Section 7 - Reproducability** in the paper. If you would like to run the code to reproduce the preprocessing, training, prediction, or API implementation, please follow the following steps.

**1. Clone the repository to your local machine.**

```bash
git clone https://github.com/kyleosung/DS3ML3_Final_Project.git
```

**2. Install the required dependencies.**

You may wish to do this inside a virtual environment.

```bash
pip install -r installation/requirements.txt
```

To train on a GPU (if available), please install Nvidia Cuda Toolkit. This software is available here: [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit).

**3.  Download the dataset.**

To do this, please navigate to the [University of Toronto Computational Social Science Lab's Chess Database](https://csslab.cs.toronto.edu/datasets/#monthly_chess_csv) and click the links corresponding to the January dataset.

**4. Unzip the Dataset and Move the Dataset**

Please unzip the data in a way compatible with your machine. You may be able to do this using WinRAR or using the Command Line, but please note that the sheer size of the dataset may make this process time-intensive.

Please move each of the dataset files into a folder outside the repository in the local path `../Data/`.

```bash
mkdir ../Data
```

**5. Parse the Dataset** 

Please run sequentially the bash scripts in repository.

```bash
chmod +x scripts/cutter.sh
./scripts/cutter.sh
chmod +x scripts/splitter.sh
./scripts/splitter.sh
```

Again, this process may be time-intensive (took approximately sixty minutes to run on my machine).

**6. Train the Dataset**

If you wish to reproduce the training, please run the main training file for the latest model (chess_SL_E8_lib)[https://github.com/kyleosung/DS3ML3_Final_Project/blob/main/chess_SL_E8_lib.py].

My training involved running the model for a couple dozen epochs on a subset of the data, training for approximately 24 hours (different amounts based on each model trained). This may vary depending on your computing power.

**7. Activate Lichess API**

Please follow the instructions given in the [Lichess Bot API](https://github.com/lichess-bot-devs/lichess-bot).


**8. Enjoy a Great Game of Chess!**

