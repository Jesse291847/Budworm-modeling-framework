import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


def set_mpl_settings():
    #Latex style font
    # mpl.rc('font', family = 'serif', serif = 'cmr10')
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'


    #Figure Size
    plt.rcParams["figure.figsize"] = [12,6]
    plt.tight_layout()

    #Font sizes
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 18, 22, 26
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('text')#, usetex=True)
    

    #Make transparent background
    plt.rc('figure', facecolor='none')
    plt.rc('axes', facecolor='none')

    #Remove axes top right
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

    #Increase linewidths
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['lines.linewidth'] = 2.25
    mpl.rcParams["legend.edgecolor"] = '#C18203'
    mpl.rcParams["legend.frameon"] = True

    #Use Custom Color Scheme
    gold_black = ('#C18203','#1C110A','#880D1E','#BFDBF7','#A663CC')
    mpl.rcParams['axes.prop_cycle'] = cycler(color= gold_black)

    #Set dpi to retina
    mpl.rc("figure", dpi=330) 


