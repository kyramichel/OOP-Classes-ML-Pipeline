import pickle
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class DataLoader:
    def load(self, path, target_col, test_size=0.2, random_state=42):
        df = pd.read_csv(path)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y,
                                 test_size=test_size,
                                 random_state=random_state)

class ModelWrapper(ABC):
    @abstractmethod
    def train(self, X, y): pass

    @abstractmethod
    def evaluate(self, X, y): pass

    @abstractmethod
    def predict(self, X): pass

    @abstractmethod
    def save(self, path): pass

    @abstractmethod
    def load(self, path): pass

class SklearnModel(ModelWrapper):
    def __init__(self, estimator=None):
        self.estimator = estimator or LogisticRegression()

    def train(self, X, y):
        self.estimator.fit(X, y)

    def evaluate(self, X, y):
        return self.estimator.score(X, y)

    def predict(self, X):
        return self.estimator.predict(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.estimator, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.estimator = pickle.load(f)

class MLPipeline:
    def __init__(self):
        self.dl = DataLoader()
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def cmd_load_data(self, csv_path, target_col):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.dl.load(csv_path, target_col)
            print("Data loaded:")
            print("  train:", self.X_train.shape)
            print("  test: ", self.X_test.shape)
        except Exception as e:
            print("Error loading data:", e)

    def cmd_train(self, model_type='logistic', **params):
        if self.X_train is None:
            print("ERROR: No data loaded. Run `load_data` first.")
            return
        if model_type == 'logistic':
            est = LogisticRegression(**params)
        else:
            print(f"ERROR: Unknown model_type '{model_type}'.")
            return
        self.model = SklearnModel(est)
        try:
            self.model.train(self.X_train, self.y_train)
            print("Model trained.")
        except Exception as e:
            print("Error training model:", e)

    def cmd_evaluate(self):
        if self.model is None:
            print("ERROR: No model trained or loaded.")
            return
        try:
            acc = self.model.evaluate(self.X_test, self.y_test)
            print(f"Accuracy: {acc:.4f}")
        except Exception as e:
            print("Error during evaluation:", e)

    def cmd_save_model(self, path):
        if self.model is None:
            print("ERROR: No model trained or loaded.")
            return
        try:
            self.model.save(path)
            print("Model saved to", path)
        except Exception as e:
            print("Error saving model:", e)

    def cmd_load_model(self, path):
        try:
            self.model = SklearnModel()
            self.model.load(path)
            print("Model loaded from", path)
        except Exception as e:
            print("Error loading model:", e)

    def cmd_predict(self, sample_csv):
        if self.model is None:
            print("ERROR: No model trained or loaded.")
            return
        try:
            df = pd.read_csv(sample_csv)
            preds = self.model.predict(df)
            print("Predictions:", preds)
        except Exception as e:
            print("Error during prediction:", e)

    
    def run(self):
        print("Welcome to MLPipeline! Type `help` for commands.")
        while True:
            line = input("> ")

            # 1) skip totally empty/whitespace lines
            if not line.strip():
                continue

            parts = line.split()
            cmd, *args = parts

            if cmd == "help":
                print(
                    "Commands:",
                    "\n  load_data <csv_path> <target_col>",
                    "\n  train [model_type]",
                    "\n  evaluate",
                    "\n  save_model <path>",
                    "\n  load_model <path>",
                    "\n  predict <sample_csv>",
                    "\n  exit / quit",
                )

            elif cmd == "load_data":
                if len(args) != 2:
                    print("Usage: load_data <csv_path> <target_col>")
                else:
                    self.cmd_load_data(*args)

            elif cmd == "train":
                if len(args) > 1:
                    print("Usage: train [model_type]")
                else:
                    model_type = args[0] if args else "logistic"
                    self.cmd_train(model_type)

            elif cmd == "evaluate":
                self.cmd_evaluate()

            elif cmd == "save_model":
                if len(args) != 1:
                    print("Usage: save_model <path>")
                else:
                    self.cmd_save_model(args[0])

            elif cmd == "load_model":
                if len(args) != 1:
                    print("Usage: load_model <path>")
                else:
                    self.cmd_load_model(args[0])

            elif cmd == "predict":
                if len(args) != 1:
                    print("Usage: predict <sample_csv>")
                else:
                    self.cmd_predict(args[0])

            elif cmd in ("exit", "quit"):
                print("Goodbye!")
                break

            else:
                print(f"Unknown command: '{cmd}'.  Type `help`.")

if __name__ == "__main__":
    MLPipeline().run()
