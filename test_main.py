"""
Test goes here, for four functions

"""

from main import *


def test_load_iris_from_path():
    path = "./Iris.csv"
    iris_data = load_iris_from_path(path)
    assert isinstance(iris_data, pd.DataFrame), f"load failed"
    assert len(iris_data) == 150, f"data loaded is wrong"


def test_desc_iris():
    # Create a sample DataFrame for testing
    sample_data = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9, 4.7],
            "sepal width (cm)": [3.5, 3.0, 3.2],
            "petal length (cm)": [1.4, 1.3, 1.5],
            "petal width (cm)": [0.2, 0.4, 0.3],
            "target": [0, 1, 2],
        }
    )
    description = desc_iris(sample_data)
    assert isinstance(description, pd.DataFrame), f"data is not the format of Dataframe"
    assert len(description) == 8, f"lost the key summary of the data"


def test_visual_iris():
    sample_data = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9, 4.7],
            "sepal width (cm)": [3.5, 3.0, 3.2],
            "petal length (cm)": [1.4, 1.3, 1.5],
            "petal width (cm)": [0.2, 0.4, 0.3],
            "target": [0, 1, 2],
        }
    )
    path = visual_iris(sample_data)
    import os

    # delete the file
    os.remove(path)
    assert path == "iris_comparison.png", f"visualization failed or file path is wrong"
