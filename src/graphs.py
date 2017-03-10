import os, fileio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def graph_returns():
    plt.figure()
    plt.title('Performance')
    plt.ylim(10, 2000)
    plt.xlim(0, 500)
    plt.yscale('log')
    plt.ylabel('Steps per episode')
    plt.xlabel('Number of episodes')
        
    for fname in os.listdir("."):
        if 'results_' not in fname or '.txt' not in fname: continue
        
        y = fileio.read_line_list(fname)
                
        wsize = 30
        window = np.ones(int(wsize))/float(wsize)
        y_av = np.convolve(y, window, 'same')
        
        plt.plot([i for i in range(len(y))][:-wsize], y_av[:-wsize], label=fname.split("results_")[1][:-4]) #_av[:-50])
        
    plt.legend()
    plt.savefig('steps.png', bbox_inches='tight')
    
    plt.figure()
    plt.title('Performance')
    #plt.ylim(10, 2000)
    wsize = 30
    plt.ylim(-100, 0)
    plt.xlim(0+wsize, 500-wsize)
    #plt.yscale('log')
    plt.ylabel('Total return')
    plt.xlabel('Number of episodes')
        
    for fname in os.listdir("."):
        if 'returns_' not in fname or '.txt' not in fname: continue
        
        y = fileio.read_line_list(fname)
                
        window = np.ones(int(wsize))/float(wsize)
        y_av = np.convolve(y, window, 'same')
        
        plt.plot([i for i in range(len(y))][wsize:-wsize], y_av[wsize:-wsize], label=fname.split("returns_")[1][:-4]) #_av[:-50])
        
    plt.legend()
    plt.savefig('returns.png', bbox_inches='tight')

graph_returns()