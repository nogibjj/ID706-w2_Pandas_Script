"""
Main cli or app entry point
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def generate_iris():
    iris = load_iris()
    data_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target = pd.DataFrame(data=iris.target, columns=["target"])
    df = pd.concat([data_iris, target], axis=1)
    df.to_csv("./Iris.csv", index=False)


def load_iris_from_path(path):
    iris_data = pd.read_csv(path)
    return iris_data


def desc_iris(data_iris):
    print(data_iris.describe())
    return data_iris.describe()


def visual_iris(data_iris):
    _, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].hist(data_iris["sepal length (cm)"], bins=10, edgecolor="black")
    axs[0, 0].set_xlabel("Sepal Length (cm)")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].set_title("Histogram of Sepal Length")

    axs[0, 1].scatter(
        data_iris["sepal length (cm)"],
        data_iris["sepal width (cm)"],
        c=data_iris.target,
        cmap="viridis",
    )
    axs[0, 1].set_xlabel("Sepal Length (cm)")
    axs[0, 1].set_ylabel("Sepal Width (cm)")
    axs[0, 1].set_title("Scatter Plot of Sepal Length vs. Sepal Width")

    axs[1, 0].boxplot(
        data_iris[data_iris.columns[:-1]].values, labels=data_iris.columns[:-1]
    )
    axs[1, 0].set_xlabel("Features")
    axs[1, 0].set_ylabel("Values")
    axs[1, 0].set_title("Box Plot of Iris Dataset")
    axs[1, 0].set_xticklabels(data_iris.columns[:-1], rotation=45)

    target_counts = data_iris["target"].value_counts()
    axs[1, 1].pie(
        target_counts, labels=target_counts.index, autopct="%1.1f%%", startangle=90
    )
    axs[1, 1].set_title("Distribution of Iris Species")

    plt.tight_layout()
    plt.savefig("iris_comparison.png")
    plt.show()

    return "iris_comparison.png"


# if __name__ == "__main__":
#     # pylint: disable=no-value-for-parameter
#     generate_iris()
#     cur_path = "./Iris.csv"
#     data = load_iris_from_path(cur_path)
#     desc_iris(data)
#     visual_iris(data)
