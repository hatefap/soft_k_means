# do not use this implementation in production setup, instead use Spark MLLib API
import findspark
findspark.init('/home/hatef/spark/')
from pyspark import SparkConf, SparkContext
from sklearn.datasets import make_blobs
import os
import numpy as np
import math
import matplotlib.pyplot as plt

# number of centers
K = 5
# dimension of data
# do not change this, beacause we want to plot the results 
D = 2
# number of samples
N = 1000


def parse_line(line):
    points = line.split(',')
    return (float(points[0]), float(points[1]))
    
def distance(u, v):
    diff = u - v
    return diff.dot(diff)
    
def compute_responsibility(pair):
    beta = 0.1
    point = np.array(pair[0])
    centroids = pair[1]
    resp = []

    total_sum = np.sum( np.exp(-beta*distance(c, point)) for c in centroids )
    
    for c in range(len(centroids)):
        centroid = centroids[c]
        dis = distance(point, centroid)
        resp.append( np.exp(-beta*dis) / total_sum )
    
    result = []
    
    for r in range(len(resp)):
        result.append((r, (resp[r]*point, resp[r])))
    return result  


def specify_cluster(pntcntrds):
    point = pntcntrds[0]
    centroids = pntcntrds[1]
    
    distances = [distance(point, c) for c in centroids]
    
    return (point, distances.index(min(distances)))
    
    
def soft_k_means(centers, max_iter=40):
    conf = SparkConf().setMaster("local[*]").setAppName("SoftKMeans")
    sc = SparkContext.getOrCreate(conf=conf)
    
    lines = sc.textFile('data.txt')
    data = lines.map(parse_line)
    
    current_centroids = np.array(data.takeSample(False, centers))
    for i in range(max_iter):
        pairing = data.map(lambda p: (p, current_centroids))
        
        responsibility = pairing.flatMap(compute_responsibility)
        
        
        new_centroids = responsibility.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).map(lambda x: x[1][0]/x[1][1])
        current_centroids = new_centroids.collect()
    
    final_centroids = current_centroids
    results = data.map(lambda p: (p, final_centroids)).map(specify_cluster).collect()
    
    x, y, labels = [],[],[]
    for p,l in results:
        x.append(p[0])
        y.append(p[1])
        labels.append(l)

    plt.scatter(x, y, c=labels)
    plt.show()
    
def main():
    if not os.path.exists('data.txt'):
        data = make_blobs(n_samples=N, n_features=D, centers=K, random_state=101)[0]
        with open('data.txt', 'w') as f:
            for p in data:
                line = ','.join([str(i) for i in p])
                f.write("{}\n".format(line))
        
    soft_k_means(K)
    
    



if __name__ == '__main__':
    main()
