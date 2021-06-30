import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D
from numpy.random import randn
from matplotlib.legend_handler import HandlerPatch

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

class AnyObject(object):
    pass

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                   edgecolor='black', hatch='xx', lw=3,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def intro():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()
    #--------------------------------------------
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
    plt.show()
    #--------------------------------------------
    t = np.arange(0., 5., 0.2)
    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    plt.show()
    #---------------------------------------------
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    plt.subplot(212)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    plt.show()
    #----------------------------------------------
    ax = plt.subplot(111)
    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2 * np.pi * t)
    line, = plt.plot(t, s, lw=2)
    plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.ylim(-2, 2)
    plt.show()
    #--------------------------------------------
    np.random.seed(19680801)
    y = np.random.normal(loc=0.5, scale=0.4, size=1000)
    y = y[(y > 0) & (y < 1)]
    y.sort()
    x = np.arange(len(y))
    plt.figure(1)
    plt.subplot(221)
    plt.plot(x, y)
    plt.yscale('linear')
    plt.title('linear')
    plt.grid(True)
    plt.subplot(222)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('log')
    plt.grid(True)
    plt.subplot(223)
    plt.plot(x, y - y.mean())
    plt.yscale('symlog', linthresh=0.01)
    plt.title('symlog')
    plt.grid(True)
    plt.subplot(224)
    plt.plot(x, y)
    plt.yscale('logit')
    plt.title('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
    plt.show()

def text():
    fig = plt.figure()
    fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('axes title')
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
    ax.text(3, 8, 'boxed italics text in data coords', style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
    ax.text(3, 2, u'unicode: Institut f\374r Festk\366rperphysik')
    ax.text(0.95, 0.01, 'colored text in axes coords',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='green', fontsize=15)
    ax.plot([2], [1], 'o')
    ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),arrowprops=dict(facecolor='black', shrink=0.05))
    ax.axis([0, 10, 0, 10])
    plt.show()

def math_text():
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2 * np.pi * t)
    plt.plot(t, s)
    plt.title(r'$\alpha_i > \beta_i$', fontsize=20)
    plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
    plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',fontsize=20)
    plt.xlabel('time (s)')
    plt.ylabel('volts (mV)')
    plt.show()

def legend():
    red_patch = mpatches.Patch(color='red', label='The red data')
    plt.legend(handles=[red_patch])
    plt.show()
    #________________________
    blue_line = mlines.Line2D([], [], color='blue', marker='*',markersize=15, label='Blue stars')
    plt.legend(handles=[blue_line])
    plt.show()
    #________________________
    plt.subplot(211)
    plt.plot([1, 2, 3], label="test1")
    plt.plot([3, 2, 1], label="test2")
    # Place a legend above this subplot, expanding itself to
    # fully use the given bounding box.
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.subplot(223)
    plt.plot([1, 2, 3], label="test1")
    plt.plot([3, 2, 1], label="test2")
    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    #_________________________
    line1, = plt.plot([1, 2, 3], label="Line 1", linestyle='--')
    line2, = plt.plot([3, 2, 1], label="Line 2", linewidth=4)
    first_legend = plt.legend(handles=[line1], loc=1)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[line2], loc=4)
    plt.show()
    #_________________________
    line1, = plt.plot([3, 2, 1], marker='o', label='Line 1')
    line2, = plt.plot([1, 2, 3], marker='o', label='Line 2')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()
    #_________________________
    z = randn(10)
    red_dot, = plt.plot(z, "ro", markersize=15)
    # Put a white cross over some of the data.
    white_cross, = plt.plot(z[:5], "w+", markeredgewidth=3, markersize=15)
    plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
    plt.show()
    #_________________________
    plt.legend([AnyObject()], ['My first handler'],
               handler_map={AnyObject: AnyObjectHandler()})
    plt.show()
    #_________________________
    c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="green",
                        edgecolor="red", linewidth=3)
    plt.gca().add_patch(c)
    plt.legend([c], ["An ellipse, not a rectangle"],
               handler_map={mpatches.Circle: HandlerEllipse()})
    plt.show()
    #_________________________

def main():
    intro()
    text()
    math_text()
    legend()

if __name__ == '__main__':
    main()



