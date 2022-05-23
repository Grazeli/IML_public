import matplotlib.pyplot as plt


def heatmap(matrix, title, path):
    # Plot heatmap of matrix comparing models based on Wilxocon P-values with cross-validation folds
    plt.imshow(matrix, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title(title)
    plt.savefig(path)