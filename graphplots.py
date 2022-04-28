import os
import matplotlib.pyplot as plt

def print_plot(array_x, array_y, name, xlabel, ylabel, out_filename):
    fig = plt.figure()
    std_scaling = 1
    plt.plot(array_x, array_y)

    #plt.legend()
    #plt.xlim(min(array_x), max(array_x))
    #plt.ylim(min(array_y), max(array_y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.grid(True)

    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig("plots/"+out_filename+"_plot.png")
    plt.savefig("plots/"+out_filename+"_plot.pdf")

    plt.close()

    print("The plot is drawn and available in the plots/ folder")
