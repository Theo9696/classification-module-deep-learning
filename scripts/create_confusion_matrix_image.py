import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    columns = ["bg", "ch", "cl", "cc", "cw", "fh", "lsb", "m", "sm", "sp", "sfc", "sb"]
    df_confusion = pd.DataFrame(data=df_confusion, index=columns, columns=columns)
    plt.matshow(df_confusion, cmap=cmap)
    # plt.title("ResNet18 Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()


if __name__ == "__main__":
    resnet = np.array([[41, 0, 0, 0, 0, 1, 8, 0, 0, 0, 0, 0],
                       [0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 57, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 125, 0, 5, 0, 0, 7, 3, 0, 0],
                       [1, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 4, 74, 0, 0, 0, 1, 0, 1],
                       [25, 0, 0, 0, 1, 0, 130, 0, 1, 0, 0, 1],
                       [0, 0, 0, 0, 3, 0, 1, 46, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 85, 2, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 34, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 100, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79]
                       ])
    cnn5 = np.array([[28, 0, 0, 0, 0, 1, 14, 0, 0, 0, 0, 0],
                     [0, 75, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                     [0, 2, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 119, 0, 0, 0, 0, 0, 0, 1, 0],
                     [2, 0, 1, 0, 33, 1, 2, 0, 1, 0, 1, 3],
                     [0, 1, 0, 0, 0, 78, 0, 1, 1, 0, 1, 1],
                     [33, 0, 0, 0, 1, 0, 123, 0, 4, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 3],
                     [2, 1, 0, 2, 0, 0, 0, 0, 87, 4, 0, 0],
                     [0, 0, 0, 2, 0, 0, 0, 0, 0, 38, 0, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 98, 1],
                     [2, 0, 0, 1, 1, 0, 0, 3, 0, 0, 0, 76]
                     ])
    plot_confusion_matrix(cnn5)
