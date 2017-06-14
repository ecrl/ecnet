import ecnet_server
import numpy as np
from sklearn import linear_model

server = ecnet_server.Server()
server.import_data()

regr = linear_model.LinearRegression()
regr.fit(server.data.learn_x, server.data.learn_y)

rmse = np.sqrt(np.mean((regr.predict(server.data.x) - server.data.y)**2))

print("RMSE: %.2f" % rmse)
