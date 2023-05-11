import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():

	def __init__(self,alpha,it,normalizacion=False):
		self.alpha = alpha
		self.it = it
		self.normalizacion = normalizacion


	def fit_dg(self,x,y):
		self.x = x
		self.y = y
		self.m = self.x.shape[0]
		if self.normalizacion:
			self.normalizacionDeCaracteristicas()
		self.x = np.concatenate((np.ones((self.m,1)),self.x), axis=1)
		self.n = self.x.shape[1]
		self.J_History = []
		self.T_History = []
		self.theta = np.zeros(self.n)
		for i in range(self.it):
			self.theta -= (self.alpha / self.m)*(np.dot(self.x, self.theta) - self.y).dot(self.x)
			self.J_History.append(self.funcionDeCoste(self.theta))
			self.T_History.append(self.theta.tolist())


	def funcionDeCoste(self, theta):
		h = np.dot(self.x,theta)
		J = (1/(2 * self.m)) * np.sum(np.square(h - self.y))
		return J


	def normalizacionDeCaracteristicas(self):
		self.mean = np.mean(self.x, axis=0)
		self.sigma = np.std(self.x, axis=0)
		self.x = (self.x-self.mean)/self.sigma


	def predict(self,x):
		x = (x-self.mean)/self.sigma
		x = np.insert(x,0,1)
		print(x,' -> ',np.dot(x,self.theta))
		fig, axs = plt.subplots(round(self.n/3), 3, figsize=(6, 4))
		col = 0
		fil = 0
		for j in range(self.n-1):
			d = np.vstack(np.arange(start=np.min(self.x[:,j+1]),stop=np.max(self.x[:,j+1]),step=0.1))
			axs[fil,col].plot(self.x[:,j+1],self.y,'gx')
			axs[fil,col].plot(d,self.theta[0]+d*self.theta[j+1],'r-')
			axs[fil,col].plot(x[j+1],self.theta[0]+x[j+1]*self.theta[j+1],'bo',linewidth=2, markersize=6)
			col += 1 
			if col==3 or col==6:
				fil += 1
				col = 0	
		plt.show()


	def graficaCosteInteraciones(self):
		plt.plot(np.arange(0,self.it),self.J_History)
		plt.xlabel('Iteraciones')
		plt.ylabel('Coste')
		plt.show()


	def graficaMultiCaracteristicas(self):
		fig, axs = plt.subplots(round(self.n/3), 3, figsize=(6, 4))
		col = 0
		fil = 0
		for j in range(self.n-1):
			d = np.vstack(np.arange(start=np.min(self.x[:,j+1]),stop=np.max(self.x[:,j+1]),step=0.1))
			axs[fil,col].plot(self.x[:,j+1],self.y,'x')
			axs[fil,col].plot(d,self.theta[0]+d*self.theta[j+1],'-')
			col += 1 
			if col==3 or col==6:
				fil += 1
				col = 0	
		plt.show()


	def graficos(self):
		self.graficaCosteInteraciones()
		self.graficaMultiCaracteristicas()


if __name__ == '__main__':
	data = np.loadtxt('Lab07/dataFutbol.txt',delimiter=",")
	x = data[:,:5]
	y = data[:,5]
	modelo = LinearRegression(0.01,1500,True)
	modelo.fit_dg(x,y)
	print('Coste minimo:',np.min(modelo.J_History))
	modelo.graficos()
	x_p = [20,10,6,2,2]
	modelo.predict(x_p)
