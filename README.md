# DS3ML3_Final_Project

Created by: Kyle Sung

Hi there! This is the main repository for my Final Project for Datasci 3ML3 in Winter 2024. Our goal was to create a Chess AI using machine learning techniques developed in the course.

**First**: If you haven't did read the paper, please do this first. It can be found in the submission box or by navigating to the portion of the repository here: [link tbd]

**Next**: If you haven't tried playing agaginst the bot, please do this next. Follow the instructions in the paper.

#### Reproducability

This next section follows **Section 6 - Reproducability** in the paper. If you would like to run the code to reproduce the preprocessing, training, prediction, or API implementation, please follow the following steps.

**1. Clone the repository to your local machine.**

```bash
git clone https://github.com/kyleosung/DS3ML3_Final_Project.git
```

**2. Install the required dependencies. [NOTE TO SELF: CREATE THE DEPENDENCIES FILE]**

You may wish to do this inside a virtual environment.

```bash
pip install installation/requirements.txt
```
  
**2.  Download the dataset.**

To do this, please navigate to the [University of Toronto Computational Social Science Lab's Chess Database](https://csslab.cs.toronto.edu/datasets/#monthly_chess_csv) and click the links corresponding to each of the datasets January through December.

**3. Unzip the Dataset and Move the Dataset**

Please unzip the data in a way compatible with your machine. You may be able to do this using WinRAR or using the Command Line, but please note that the sheer size of the dataset may make this process time-intensive.

Please move each of the dataset files into a folder outside the repository in the local path `../DataTrain`.

**4. Parse the Dataset [NOTE TO SELF: ADD THESE TO REPO AND CHANGE LOCAL PATHS]** 

Please run sequentially the bash scripts in repository.

```bash
chmod +x splitter.sh
./splitter.sh
chmod +x cutter.sh
./cutter.sh
```

Again, this process may be time-intensive (took approximately sixty minutes to run on my machine).

**5. Train the Dataset**

If you wish to reproduce the training, please run the main training file for the latest model, titled ``NOTE TO SELF: ADD TITLE!`` (possibly add link too).

My training involved running the model for 25 epochs on a subset of the data, training for approximately 24 hours. This may vary depending on your computing power.

**6. Activate Lichess API**



**7. Enjoy a Great Game of Chess!**

