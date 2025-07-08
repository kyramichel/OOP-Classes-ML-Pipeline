
# OOP-Classes-ML-Pipeline

A demonstration of an object-oriented machine learning pipeline using classes.

This project demonstrates an object-oriented machine learning pipeline using classes. It uses `MLClassificationPipeline.py` to:

1. Load a CSV
2. Train a `LogisticRegression` model
3. Evaluate accuracy
4. Save / load the model
5. Predict new samples

We’ve included `iris_small.csv` (6 rows, 4 features + species) for demonstration purposes.

## Prerequisites

- Python 3.7+
- `pandas` (version 1.3.0+)
- `scikit-learn` (version 0.24.0+)

You can install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn
```

## Project Structure

```
OOP-Classes-ML-Pipeline/
├── MLClassificationPipeline.py
├── iris_small.csv
└── README.md
```

## Files

- **MLClassificationPipeline.py** — the interactive REPL pipeline
- **iris_small.csv** — tiny example dataset

Contents of `iris_small.csv` (header + 6 rows):

```
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
6.2,3.4,5.4,2.3,virginica
5.9,3.0,5.1,1.8,virginica
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
```

## Running the Demo

1. Launch the script:

   ```bash
   python MLClassificationPipeline.py
   ```

2. You’ll see:

   ```
   Welcome to MLPipeline! Type `help` for commands.
   >
   ```

3. Load the data:

   ```bash
   > load_data iris_small.csv species
   Data loaded:
     train: (4, 4)
     test:  (2, 4)
   >
   ```

4. Train:

   ```bash
   > train
   Model trained.
   >
   ```

5. Evaluate:

   ```bash
   > evaluate
   Accuracy: 1.0000
   >
   ```

6. Save your model:

   ```bash
   > save_model iris_model.pkl
   Model saved to iris_model.pkl
   >
   ```

7. (Optional) Exit and relaunch, then reload:

   ```bash
   > exit
   Goodbye!
   $ python MLClassificationPipeline.py
   Welcome to MLPipeline! Type `help` for commands.
   > load_model iris_model.pkl
   Model loaded from iris_model.pkl
   >
   ```

8. Predict new samples from a CSV (`new.csv`—you can create your own):

   ```bash
   > predict new.csv
   Predictions: ['setosa' 'virginica']
   >
   ```

9. Quit:

   ```bash
   > quit
   Goodbye!
   ```

Next, I will be extending the pipeline to support more features and functionalities.
