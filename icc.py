from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg as LA
import math  

class ICC():
  i=0
  def __init__(self):
        self.x =None
        self.u =None
        self.ii=None
        self.n=None
        self.c=None
        self.clusterscentroids=[]
        self.clusters_centroids=None
        self.diffarray=[]
        self.summarraySbe=[]


  def fit(self,x,u):
       self.x =x
       self.u =u
       self.n =x.shape[0]
       self.c =u.shape[1]
       self.clusters_centroidsfn()
       return self.iccc()

  def iccc(self): 

        return (self.Sbe()/self.n)*self.Dmin()*(math.sqrt(self.c))

  def m(self):
    #dataset centroid
    #its neccesary that x be a matrix (only works for multivariate case)
        return np.sum(self.x,axis=0)/self.x.shape[0]

  def mei(self,i):
    #This function calculate the partition centroid
    #x is the dataset matrix
    # u is the matrix of fuzzy membership and i is the ith cluster vector
        return np.sum(self.x*self.u[:,self.i][:, np.newaxis],axis=0)/np.sum(self.u[:,i])

  def clusters_centroidsfn(self):
        for i in range(self.u.shape[1]):
              self.clusterscentroids.append(self.mei(i))
        self.clusters_centroids=np.asarray(self.clusterscentroids) 

  def Dmin(self):
   #computes the minimun euclidean distance between all the cluster centers
           
        for i in range(len(self.clusters_centroids)-1):
            self.diffarray.extend(euclidean_distances(self.clusters_centroids[(i+1):,:],self.clusters_centroids[i,:].reshape(1, -1)))
        return min(np.asarray(self.diffarray))
  
  def Sbe(self):
        for i in range(self.clusters_centroids.shape[0]):
              self.summarraySbe.append(sum(u[:,i]*LA.norm(self.clusterscentroids[i]-m(x), 2)))
        return sum(self.summarraySbe)

