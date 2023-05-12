import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression():

	def __init__(self,alpha,it,normalizacion=False):
		self.alpha = alpha
		self.it = it
		self.normalizacion = normalizacion


	def fit_dg(self,x,y):
		self.x = np.concatenate((x,np.square(x)),axis=1)
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


	def fit_NormalEquation(self,x,y):
		self.x = np.concatenate((x,np.square(x)),axis=1)
		self.y = y
		self.m = self.x.shape[0]
		if self.normalizacion:
			self.normalizacionDeCaracteristicas()
		self.x = np.concatenate((np.ones((self.m,1)),self.x), axis=1)
		self.n = self.x.shape[1]
		self.theta = np.dot(np.dot(np.linalg.inv(np.dot(self.x.T,self.x)),self.x.T),self.y)


	def funcionDeCoste(self, theta):
		h = np.dot(self.x,theta)
		J = (1/(2 * self.m)) * np.sum(np.square(h - self.y))
		return J


	def normalizacionDeCaracteristicas(self):
		self.mean = np.mean(self.x, axis=0)
		self.sigma = np.std(self.x, axis=0)
		self.x = (self.x-self.mean)/self.sigma


	def predict(self,x):
		x = np.array(x)
		x = np.concatenate((x,np.square(x)),axis=0)
		x = (x-self.mean)/self.sigma
		x = np.insert(x,0,1)
		print("Resultado:",np.dot(x,self.theta))
		d = np.vstack(np.arange(start=np.min(self.x[:,2]),stop=np.max(self.x[:,2]),step=0.1))
		plt.plot(self.x[:,2],self.y,'x')
		#plt.plot(d,self.theta[0]+np.square(d)*self.theta[2],'-')
		plt.plot(x[1],np.dot(x,self.theta),'o')
		plt.xlabel('w2')
		plt.ylabel('y')
		plt.show()


	def graficaCosteInteraciones(self):
		plt.plot(np.arange(0,self.it),self.J_History)
		plt.xlabel('Iteraciones')
		plt.ylabel('Coste')
		plt.show()


	def grafica2D(self):
		d = np.vstack(np.arange(start=-1,stop=1,step=0.1))
		plt.plot(self.x[:,1],self.y,'x')
		plt.plot(d,self.theta[0]+np.square(d)*self.theta[2],'-')
		plt.xlabel('w2')
		plt.ylabel('y')
		plt.show()


	def graficos(self):
		self.graficaCosteInteraciones()
		self.grafica2D()


if __name__ == '__main__':
	data = np.loadtxt('Lab08/d1.txt',delimiter=",")
	x = data[:,:-1]
	y = data[:,-1]
	modelo = PolynomialRegression(0.03,4000,True)
	modelo.fit_dg(x,y)
	modelo.graficos()
	print('Coste minimo:',np.min(modelo.J_History))
	x_p = [40]
	modelo.predict(x_p)
