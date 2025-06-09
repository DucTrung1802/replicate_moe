import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:

    def __init__(self):
        pass

    def draw_lines(
        self,
        x: np.ndarray,
        *ys: np.ndarray,
        labels: list[str] = None,
        title: str = "Title",
        xlabel: str = "Epoch",
        ylabel: str = "Value",
    ):
        # Validate input lengths
        for i, y in enumerate(ys):
            if len(x) != len(y):
                raise ValueError(
                    f"Length mismatch: x has length {len(x)}, but y[{i}] has length {len(y)}"
                )

        # Validate labels
        if labels is None:
            labels = [f"Series {i+1}" for i in range(len(ys))]
        elif len(labels) != len(ys):
            raise ValueError(f"Expected {len(ys)} labels, got {len(labels)}")

        # Plotting
        plt.figure(figsize=(10, 5))
        for y, label in zip(ys, labels):
            plt.plot(x, y, label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def read_csv_and_draw(self, path: str, title: str = None):
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {path}")

        if "epoch" not in df.columns:
            raise ValueError("CSV file must contain an 'epoch' column")

        x = df["epoch"].values
        ys = [df[col].values for col in df.columns if col != "epoch"]
        labels = [col for col in df.columns if col != "epoch"]

        self.draw_lines(x, *ys, labels=labels, title=title)


if __name__ == "__main__":

    # Example usage
    visualizer = Visualizer()
    visualizer.read_csv_and_draw(
        path="visualizer/ckpt_Nornal_MobileNetV2_early_stopping_patience_10_acc_80.99.csv",
        title="Normal MobileNetV2 Early Stopping (Patience 10, Acc 80.99)",
    )
