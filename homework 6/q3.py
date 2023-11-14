import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd 

X = np.array([1,2,3,4])
Y = np.array([2.2, 4.1, 6.3, 7.9])

kernel_x = np.arange(-len(X),len(X), 0.1)
gamma = 0.5

# Gaussian function
def kernel(gamma, ker_x, xi):
    const = 1 / (gamma * np.sqrt( 2 * np.pi ))
    exp_num = -0.5 * np.square( ( xi - ker_x ))
    exp_den = gamma * gamma
    exp = exp_num / exp_den
    return const * np.exp(exp)

# Plotting the kernels
kernels = {'kernel_x': kernel_x}
for entry in X: 
    input_string= 'x_value_{}'.format(np.round(entry, 4)) 
    kernels[input_string] = kernel(gamma, kernel_x, entry)
kernels_df = pd.DataFrame(data=kernels)
y_all = kernels_df.drop(columns='kernel_x')
graph = px.line(kernels_df, x='kernel_x', y=y_all.columns, title='Gaussians', range_x=[-2,5])
graph.show()

# Calculate kernel weights
def get_weights(gamma, entry, X ): 
    w_row = []
    for x_i in X: 
        ki = kernel(gamma, x_i, entry)
        ki_sum = np.sum(kernel(gamma, X, entry))
        w_row.append(ki/ki_sum)
    return w_row

# Calculate a prediction
def y_prediction(gamma, entry, X): 
    w = get_weights(gamma, entry, X)
    prediction = np.sum(np.dot(Y,w))
    return prediction

predictions = []
for entry in X:
    prediction = y_prediction(gamma, entry, X)
    predictions.append(prediction)

'''
Calculate Y at x=2.5:
'''
print(f"At x = 2.5: {y_prediction(gamma, 2.5, X)}")

data = {'x': X, 'y': Y, 'y_manual': np.array(y_all)}
graph = px.scatter(x=X,y=Y)
graph.add_trace(go.Scatter(x=X, y=np.array(predictions), name='Manual KR',  mode='lines'))
graph.show()