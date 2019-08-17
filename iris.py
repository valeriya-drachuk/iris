### Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import show, curdoc
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.layouts import row, column
from bokeh.models.widgets import Tabs, Panel
from sklearn.linear_model import LinearRegression

### Load dataset
df = pd.read_csv('C:/Users/Lera/Documents/iris.csv', names = ['sepal_length','sepal_width','petal_length','petal_width','class'])

### Clean data
df['class'] = df['class'].replace({"Iris-": ""}, regex = True)
df['class'] = df['class'].astype('category')

### Perform basic data exploration
print(df.shape)
print(df.head())
print(df.tail())
print(df.describe())
print(df.info())

### Create new df's for species
setosa = df[df['class'] == 'setosa']
virginica = df[df['class'] == 'virginica']
versicolor = df[df['class'] == 'versicolor']

### Create basic plot with seaborn
sns.lmplot(x = 'sepal_length', y = 'sepal_width', data = df, hue = 'class')
plt.title('Classifying Iris')
plt.xlabel('sepal length in cm')
plt.ylabel('sepal width in cm')

### Create advanced plot with bokeh

### Plot1
flowers = [setosa, virginica, versicolor]



def linreg(x, y):
    slope, intercept = np.polyfit(x,y,1)
    return slope, intercept

set_x, sety = linreg(setosa['sepal_length'], setosa['sepal_width'])

slope_setosa, int_setosa = np.polyfit(setosa['sepal_length'], setosa['sepal_width'], 1)
slope_virginica, int_virginica = np.polyfit(virginica['sepal_length'], virginica['sepal_width'], 1)
slope_versicolor, int_versicolor = np.polyfit(versicolor['sepal_length'], versicolor['sepal_width'], 1)

def pred(x, slope, intercept):
    y = x * slope + intercept
    return y

x_set = np.array([4,8])
y_set_pred = pred(x_set, slope_setosa, int_setosa)

print(y_set_pred)

x_virg = np.array([4,8])
y_virg_pred = pred(x_virg, slope_virginica, int_virginica)
print(y_virg_pred)

x_vers = np.array([4,8])
y_vers_pred = pred(x_vers, slope_versicolor, int_versicolor)
print(y_vers_pred)

print('setosa slope:', slope_setosa, 'setosa intercept:', int_setosa)
print('virginica slope:', slope_virginica, 'virginica intercept:', int_virginica)
print('versicolor slope:', slope_versicolor, 'versicolor intercept:', int_versicolor)

source = ColumnDataSource(df)
hover1 = HoverTool(tooltips = [('species name','@class'), ('sepal length','@sepal_length cm'), ('sepal width', '@sepal_width cm')])

plot1 = figure(x_axis_label = 'sepal length (cm)', y_axis_label = 'sepal width (cm)', title = 'Sepal Length vs. Sepal Width',tools = [hover1])

mapper = CategoricalColorMapper(factors = ['setosa', 'virginica', 'versicolor'], palette = ['red', 'green', 'blue'])

plot1.circle('sepal_length', 'sepal_width', source = source, color =dict(field='class', transform = mapper), fill_alpha = 0.5, legend = 'class')
plot1.line(x_set, y_set_pred, color = 'red')
plot1.line(x_virg, y_virg_pred, color = 'green')
plot1.line(x_vers, y_vers_pred, color = 'blue')

### Plot2
hover2 = HoverTool(tooltips = [('species name', '@class'), ('petal length', '@petal_length cm'), ('petal width','@petal_width cm')])
mapper2 = CategoricalColorMapper(factors = ['setosa', 'virginica', 'versicolor'], palette = ['magenta', 'turquoise', 'cornflowerblue'])

plot2 = figure(x_axis_label = 'petal length (cm)', y_axis_label = 'petal width (cm)', title = 'Petal Length vs. Petal Width', tools = [hover2])
plot2.circle('petal_length', 'petal_width', source = source, color = dict(field = 'class', transform = mapper2), fill_alpha = 0.5, legend = 'class')

first = Panel(child = plot1, title = 'Sepal')
second = Panel(child = plot2, title = 'Petal')
tabs = Tabs(tabs = [first, second])

show(tabs)









