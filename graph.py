import os
import matplotlib.pyplot as plt

def print_plot(array_x, array_y, name, line_label, xlabel, ylabel, out_filename):
    fig = plt.figure()
    std_scaling = 1
    plt.plot(array_x, array_y, label=line_label)

    plt.legend()
    #plt.xlim(0), max(array_x))
    #plt.ylim(0, max(arra_y))
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
