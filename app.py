import numpy as np
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

def binary_activation(y, threshold=1):
  return 1 if y >= threshold else 0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
      weight = float(request.form.get('weights'))
      training_data_value = request.form.get('training-data-select')
      print(training_data_value)
      weights = np.array((weight, weight))
      threshold = float(request.form.get('threshold'))
        
      x1 = np.array([float(x) for x in request.form.getlist('x1[]')])
      x2 = np.array([float(x) for x in request.form.getlist('x2[]')])
      x3 = np.array([float(x) for x in request.form.getlist('x3[]')])
      x4 = np.array([float(x) for x in request.form.getlist('x4[]')])
      t = np.array([float(x) for x in request.form.getlist('t[]')])
        
      net1 = np.dot(x1, weights)
      net2 = np.dot(x2, weights)
      net3 = np.dot(x3, weights)
      net4 = np.dot(x4, weights)
      net = [net1, net2, net3, net4]
        
      fnet1 = binary_activation(net1, threshold)
      fnet2 = binary_activation(net2, threshold)
      fnet3 = binary_activation(net3, threshold)
      fnet4 = binary_activation(net4, threshold)
      fnet = [fnet1, fnet2, fnet3, fnet4]
        
      return render_template('index.html', x1=x1, x2=x2, x3=x3, x4=x4, t=t, weight=weight, threshold=threshold, net=net, fnet=fnet)
    else:
      weight = 1
      threshold = 1
      x1 = np.array((1, 1))
      x2 = np.array((1, 0))
      x3 = np.array((0, 1))
      x4 = np.array((0, 0))
      t = np.array((1, 0, 0, 0))
      return render_template('index.html', x1=x1, x2=x2, x3=x3, x4=x4, t=t, weight=weight, threshold=threshold)

if __name__ == "__main__":
  app.run(debug=True)