[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/rtXWQ2et)
# Assignment 1

## Distributed in Github Repository e4040-2023fall-assign1

The assignment is distributed as a Jupyter Notebook `Assignment 1.ipynb` and several utility files from a Github repository and/or via Courseworks. Additional instructions and support materials for the assignment can be found in file `README.md` which comes with the assignment. The assignment uses TensorFlow version 2.4.

### Detailed instructions how to submit the solution:

1. The assignment will be distributed as a Github classroom assignment - as a special repository accessed through a link
2. Student's copy of the assignment gets created automatically with a special name - <span style="color:red"><strong>students have to rename per instructions below</strong></span>
3. The solution to the assignment has to be pushed into the same repository and submitted through Gradescpe
4. Three files/screenshots need to be uploaded into the directory `figures/` which prove that the assignment has been done in the cloud

### Screenshots for illustrating/documenting that the assignment was done in the Google Cloud:

In folder `figures/`, the instructors provide 3 screenshots as an example that shows that the assignment is done in the Google Cloud:
1. A screenshot showing that a VM instance is running under the project **ecbm-UNI**.
2. A screenshot showing that a Jupyter Notebook is running in the instance under **`envTF24`**.
3. A screenshot showing the `Assignment 1.ipynb` file, with the **IP address** on top of the browser.

Students must upload three similar screenshots in the same directory, but with the modified names - add UNI in front of the file names:
1. UNI_gcp_work_example_screenshot_1.png
2. UNI_gcp_work_example_screenshot_2.png
3. UNI_gcp_work_example_screenshot_3.png

### <span style="color:red"><strong>TODO:</strong></span> (Re)naming of the student repository

***INSTRUCTIONS*** for naming the student's repository for assignments (with one student):
* Students need to use the following name for the repository with their solutions: `e4040-2023fall-assign1-UNI` (the first part "e4040-2023fall-assign1" will be inherited from the assignment repository, so only UNI needs to be changed) 
* Initially, the system may give the repository a name which ends with a student's Github user ID. The student should change that name and replace it with the name requested in the point above
  * Good Example: `e4040-2023fall-assign1-zz9999`
  * Bad example: `e4040-2023fall-assign1-e4040-2023fall-assign0-zz9999`
* This change can be done from the "Settings" tab which is located on the repository page.

***INSTRUCTIONS*** for naming the students' solution repository for assignments with more students, such as the final project (students need to use a 4-letter groupID):
* Template: `e4040-2023fall-Project-GroupID-UNI1-UNI2-UNI3`
* Example: `e4040-2023fall-Project-MEME-zz9999-aa9999-aa0000`

## Organization of Directory

```
e4040-2023fall-assign1-yh3612
├── Assignment1-intro.ipynb
├── README.md
├── figures
│   ├── yh3612_gcp_work_example_screenshot_1.jpg
│   ├── yh3612_gcp_work_example_screenshot_2.jpg
│   └── yh3612_gcp_work_example_screenshot_3.jpg
├── output.txt
├── requirements.txt
├── save_models
│   └── best_model.pkl
├── task1-basic_classifiers.ipynb
├── task2-mlp_numpy.ipynb
├── task3-mlp_tensorflow.ipynb
├── task4-questions.ipynb
└── utils
    ├── __pycache__
    │   ├── display_funcs.cpython-311.pyc
    │   ├── layer_funcs.cpython-311.pyc
    │   ├── layer_utils.cpython-311.pyc
    │   └── train_funcs.cpython-311.pyc
    ├── classifiers
    │   ├── __pycache__
    │   │   ├── basic_classifiers.cpython-311.pyc
    │   │   ├── logistic_regression.cpython-311.pyc
    │   │   ├── mlp.cpython-311.pyc
    │   │   ├── softmax.cpython-311.pyc
    │   │   └── twolayernet.cpython-311.pyc
    │   ├── basic_classifiers.py
    │   ├── logistic_regression.py
    │   ├── mlp.py
    │   ├── softmax.py
    │   └── twolayernet.py
    ├── display_funcs.py
    ├── layer_funcs.py
    ├── layer_utils.py
    └── train_funcs.py

7 directories, 30 files
```

