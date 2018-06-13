#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import pandas as pd

from matplotlib.colors import ListedColormap
from sklearn import neighbors



axis_font_size = 15
fig_size = (8, 6)
title_fontsize = 20

marker_size = 100

def temperature_conversion_plot():
    x = np.linspace(-5, 100, 100)
    y = x*(9/5) + 32

    f, ax = plt.subplots(figsize=fig_size)
    ax.plot(x, y)

    ax.set_xlabel(r'$C$, Temperature in Degrees Celcius', fontsize=axis_font_size)
    ax.set_ylabel(r'$F$, Temperature in Degrees Fahrenheit', fontsize=axis_font_size)

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))

    ax.set_title(r'Linear Function : $F = \frac{9}{5}C + 32$', fontsize=20)
    ax.title.set_position([.5, 1.05])

    ax.grid(True, which='both')

    pass

def rand_fahrenheit(x):
    return (9/5)*x + 32 + 2*np.random.randn()


def temperature_data():

    np.random.seed(24)

    x = [i + np.random.randn() for i in range(1, 50)]
    y = [rand_fahrenheit(i) for i in x]

    x = np.array(x)
    y = np.array(y)

    xlim = (min(x) - 1, max(x) + 1)
    ylim = (min(y) - 1, max(y) + 1)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    ax.set_xlabel('C, Temperature in Degrees Celcius', fontsize=15)
    ax.set_ylabel('F, Temperature in Degrees Fahrenheit', fontsize=15)
    ax.set_title(r'Temperature Data', fontsize=title_fontsize)

    ax.grid(True, which='both')


    return [x, y]

def plot_regression(x, y, b0, b1):

    x0 = min(x) - 1
    x1 = max(x) + 1
    y0 = min(y) - 1
    y1 = max(y) + 1

    xlim = (x0, x1)
    ylim = (y0, y1)

    x_line = xlim
    y_line = (x0*b1 + b0, x1*b1 + b0)

    f, ax = plt.subplots(figsize=(8,6))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    ax.set_xlabel('C, Temperature in Celcius', fontsize=axis_font_size)
    ax.set_ylabel('F, Temperature in Fahrenheit', fontsize=axis_font_size)
    ax.set_title('Model Visualization', fontsize=title_fontsize)

    ax.plot(x_line, y_line, 'k-', label='Model Predictions')
    ax.scatter(x, y, label='Values From Data')

    ax.legend()
    ax.grid(True, which='both')

    pass


def plot_predictions_vs_true(y, y_hat):

    f, ax = plt.subplots(figsize=fig_size)

    ax.scatter(y, y_hat)

    ax.set_xlabel('Measured Values', fontsize=axis_font_size)
    ax.set_ylabel('Predicted Values', fontsize=axis_font_size)
    ax.set_title('Visualization of Prediction Accuracy', fontsize=title_fontsize)
    ax.grid(True, which='both')

    pass


def plane3d_plot():

    def f(x, y):
        return 10*x + 5*y

    x = np.linspace(-50, 55, 5)
    y = np.linspace(-50, 55, 5)

    X, Y = np.meshgrid(x, y)
    Z = f(X,Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');

def bowl3d_plot():

    def f(x, y):
        return np.sin(x) + np.sin(y)

    x = np.linspace(-np.pi, np.pi, 0.2)
    y = np.linspace(-np.pi, np.pi, 0.2)

    X, Y = np.meshgrid(x, y)
    Z = f(X,Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');


def knn_make_data():
    np.random.seed(42)
    r = 20
    x1  = np.array([-0.5  + np.random.normal() for i in range(r)])
    y1  = np.array([-0.5  + np.random.normal() for i in range(r)])

    x2  = np.array([0.5  + np.random.normal() for i in range(r)])
    y2  = np.array([0.5  + np.random.normal() for i in range(r)])

    v1 = np.array([x1, y1]).T #blue
    v2 = np.array([x2, y2]).T #red

    return [x1, x2, y1, y2, v1, v2]

def knn_example(new_data = False, return_data=False, vectors=False):

    x1, x2, y1, y2, v1, v2 = knn_make_data()

    if return_data == True:
        return [v1, v2]


    # Set up plot
    f, ax = plt.subplots(figsize=fig_size)
    ax.scatter(x1, y1, c='b', marker='^', s=[marker_size for i in x1], label='Blue Triangles')
    ax.scatter(x2, y2, c='r', marker='o', s=[marker_size for i in x2], label='Red Circles')
    ax.set_xlabel('x', fontsize=axis_font_size, x=1.05)
    ax.set_ylabel('y', fontsize=axis_font_size, y=0, rotation=0)

    lim = (-3, 3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    title = 'Feature Space'

    # If new_data = True : add a point at 0, 0
    if (new_data == True):
        x3 = np.array([0])
        y3 = np.array([0])
        ax.scatter(x3, y3, c='g', marker='*', s=[marker_size*3], label='New Data')

        title = 'New Data Point (0, 0)'

        if vectors == True:
            v = np.concatenate([v1, v2])
            o = np.array([0,0]) # origin

            # Get Number of Points
            N = len(v)
            # Initialize array to hold distances
            d = np.zeros(N)

            # Loop over points to get distances
            for i in range(N):
                d[i] = np.sqrt((o - v[i]).dot(o - v[i]))

            # Get Three smallest points :

            for i in range(3):
                nearest = v[d == d.min()].flatten()
                nn_x = nearest[0]
                nn_y = nearest[1]

                ax.arrow(0,0,nn_x, nn_y, width=0.005, length_includes_head=True, color='k')

                v = v[d > d.min()]
                d = d[d > d.min()]
            lim = (-1, 1)
            title = 'K=3 Nearest Neighbors'

    # Add legend to plot (here so that it adapts to whether or not there is anew point)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title(title, fontsize=title_fontsize)
    ax.legend()

    pass

def knn_boundaries(K):

    x1, x2, y1, y2, v1, v2 = knn_make_data()

    X = np.concatenate([v1, v2])
    # Create labels
    y = [1 for i in range(len(x1))]
    y.extend([0 for i in range(len(x2))])

    h = .02  # step size in the mesh

    # Create color maps
    #0000FF
    cmap_light = ListedColormap(['#FFAAAA', '#00FFFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    #for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(K)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=fig_size)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(x1, y1, c='b', marker='^')
    plt.scatter(x2, y2, c='r', marker='o')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundaries (k = %i)"
              % (K), fontsize=title_fontsize)
    plt.xlabel('x', fontsize=axis_font_size)
    plt.ylabel('y', fontsize=axis_font_size)
    plt.show()
