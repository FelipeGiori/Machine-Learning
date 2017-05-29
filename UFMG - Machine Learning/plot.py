import numpy as np
import matplotlib.pyplot as plt


def main():
    # Reads variables
    plot1 = np.genfromtxt('results/MBGD/teste_MBGD_05_10_100.txt')
    plot2 = np.genfromtxt('results/MBGD/teste_MBGD_05_50_100.txt')
    plot3 = np.genfromtxt('results/MBGD/teste_MBGD_1_10_100.txt')
    plot4 = np.genfromtxt('results/MBGD/teste_MBGD_1_50_100.txt')
    plot5 = np.genfromtxt('results/MBGD/teste_MBGD_10_10_100.txt')
    plot6 = np.genfromtxt('results/MBGD/teste_MBGD_10_50_100.txt')
    [m] = plot1.shape
    iterations = np.arange(1, m + 1)
    plt.plot(iterations, plot1, label="MBGD = 10, learning rate = 0.5")
    plt.plot(iterations, plot2, label="MBGD = 50, learning rate = 0.5")
    plt.plot(iterations, plot3, label="MBGD = 10, learning rate = 1")
    plt.plot(iterations, plot4, label="MBGD = 50, learning rate = 1")
    plt.plot(iterations, plot5, label="MBGD = 10, learning rate = 10")
    plt.plot(iterations, plot6, label="MBGD = 50, learning rate = 10")
    plt.xlabel('Epochs')
    plt.ylabel('Training error (%)')
    plt.title('Mini-batch Gradient descent training error with 50 hidden layer units')
    legend = plt.legend(fancybox=True)
    legend.get_frame().set_alpha(0.5)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
