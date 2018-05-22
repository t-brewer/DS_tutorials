#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


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


def knn_example(testing_data = False):

    np.random.seed(42)

    r = 20

    x1 = [2  + np.random.randn() for i in range(r)]
    y1 = [2  + np.random.randn() for i in range(r)]
    A = (x1, y1)

    x2 = [4  + np.random.randn() for i in range(r)]
    y2 = [4  + np.random.randn() for i in range(r)]
    B = (x2, y2)

    f, ax = plt.subplots(figsize=fig_size)
    
    ax.scatter(x1, y1, c='b', marker='^', s=[marker_size for i in x1])
    ax.scatter(x2, y2, c='r', marker='o', s=[marker_size for i in x2])

    ax.set_xlabel('x', fontsize=axis_font_size)
    ax.set_ylabel('y', fontsize=axis_font_size)

    if testing_data == True:
        x3 = [2.5]
        y3 = [4]
        C = (x3, y3)
        
        ax.scatter(x3, y3, c='k', marker='+', s = [marker_size*2], label='New Data')

        ax.legend()

        return [A, B, C]


    pass

