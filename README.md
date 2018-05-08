# SurvivalRF

Random Forest implementation for Survival Analysis

* This repo has currently the code submitted towards the final project of STAT 689 course, Spring 2018
* We didn't actually do the complete implementation of our proposed features in project proposal
* Rather we have written minimum-working-examples or basic skeletons of the trees which can be upscaled

### Key things done

* Implement a Decision Tree (Classifier and Regressor) that can handle categorical variables as-is, without 1-hot encoding them **unlike** Scikit-learn
* Implement a Survival Tree

### Description of folders and files

* `/notebooks`- Jupyter Notebooks in Python with our skeleton implementations
* `/datasets` - Datasets (`.csv`) files for the examples in notebooks
* `/docs` and `/site` are the folders for MKdocs, i.e. for package documentation. Nothing to evaluate there now from the project perspective.

You can go to here to [documentation](https://nayanchaudhary.github.io/SurvivalRF/)
to see what we had started with as a scaffold for the documentation of our package.
This can/will be used later when we finish the package.

### Things we couldn't do that were mentioned on the project proposal

* Couldn't write code for Survival *Random Forest*. Code only up until *Decision Tree*
* Couldn't finish package level code. We evaluated 2 scenarios:
  1. Writing the complete package ground up
  
  After going through the code for *RandomForestSRC* in R and also in general through  *Scikit-Learn's * code in Python, we concluded pretty soon that this is a poor approach. There's a lot of production quality code sitting out there that we should be leveraging. Also it would be just too inefficient writing a production quality package from scratch
  
  2. Take Scikit's code and edit it just enough by changing *criterion* and *estimation* etc.
  
  This seemed like a better and efficient approach. However since the data structure for Survival is different from classification and regression (in terms of mandatory columns and their format) etc., this would involve editing a lot of functions and clasess in Scikit all the way upto its  `base ` directories. Basically we realized we'd had to make changes to many classes and for that had to go through a lot of dependency strucutre of scikit's code.
  
  In the future, a pragmatic apporach would be to copy relevant code from Scikit (to give package-level scaffold), infuse ideas from our notebooks/examples and edit code for survival analyses as well as handling categorical variables.
