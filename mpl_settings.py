import matplotlib as mpl
from pathlib import Path
from os import name as os_name
import matplotlib.pyplot as plt
import matplotlib.font_manager
# print(mpl.rcParams.keys())

# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

def fmt(x, val):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    # return r'${} \times 10^{{{}}}$'.format(a, b)
    return rf'{a}E+{b:02}'

# mpl.rcParams['lines.linestyle'] = '--'
#mpl.rcParams['legend.fontsize'] = 'large' #'x-large'
mpl.rcParams['legend.shadow'] = False
# mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 1.5
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams.update({'font.size': 18})

# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = 'Liberation Sans'
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"

if 'posix' in os_name:
    result_dir = Path(r"")
else:
    result_dir = Path(r"")
mpl.rcParams["savefig.directory"] = result_dir

"""
from matplotlib.pyplot import subplots, xlabel, ylabel, grid, show
fig, ay = subplots()

# Using the specialized math font elsewhere, plus a different font
xlabel(r"The quick brown fox jumps over the lazy dog", fontsize=18)
# No math formatting, for comparison
ylabel(r'Italic and just Arial and not-math-font', fontsize=18)
grid()

show()
"""