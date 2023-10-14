import numpy as np
from matplotlib import pyplot as plt



def find_minimum_distance(centroids, point):
    return np.argmin(np.sum((centroids - point) ** 2, axis = -1))

def k_mean_step(centroids, points, k):
    assignemnts = dict.fromkeys(list(range(k)))
    for key in assignemnts.keys():
        assignemnts[key] = list()
    for point in points:
        closets_centroid = find_minimum_distance(centroids=centroids, point=np.array(point))
        assignemnts[closets_centroid].append(point)
    for key in assignemnts.keys():
        centroids[key] = np.mean(np.array(assignemnts[key]), axis=0)
    return centroids, assignemnts

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


file_lines = ''
with open('./Points.txt', 'r', encoding='UTF8') as f:
    file_lines = f.read().split('\n')

number_of_points = int(file_lines[0])

points = list()
for i in range(number_of_points):
    x, y = map(float, file_lines[i+1].split())
    points.append((x, y))





X = np.array(points)[:, 0]
Y = np.array(points)[:, 1]
min_X, max_X = np.min(X), np.max(X)
min_Y, max_Y = np.min(Y), np.max(Y)

cmap = get_cmap(10)
area = 10
fig = plt.figure()
plt.scatter(X, Y,
            s=np.ones(shape=(X.shape[0],)) * area,
            c=cmap(0),
            alpha=0.5)

plt.savefig('res01.png')

k = 2

for count in range(2):
    centroids = np.zeros(shape=(k, 2))
    for i in range(k):
        centroids[i, 0] =  np.random.uniform(min_X, max_X)
        centroids[i, 1] =  np.random.uniform(min_Y, max_Y)
    error = float('inf')
    assignemnts = dict.fromkeys(list(range(k)))

    while(error > 1e-8):
        new_centroids, assignemnts = k_mean_step(centroids=centroids.copy(), points=points, k=k)
        error = np.sum(np.abs(centroids - new_centroids))
        centroids = new_centroids

    cmap = get_cmap(10)
    area = 10
    fig = plt.figure()
    for i in range(k):
        X = np.array(assignemnts[i])[:, 0]
        Y = np.array(assignemnts[i])[:, 1]
        print(X.shape, Y.shape)
        plt.scatter(X, Y,
                    s=np.ones(shape=(X.shape[0],)) * area,
                    c=cmap(i),
                    alpha=0.5,
                    label=f'Segment {i}')

    plt.scatter(centroids[:,0], centroids[:, 1], s=30, label='Centorids')
    plt.legend()
    plt.savefig(f'res0{count+2}.png')


mean_of_points = np.mean(np.array(points), axis=0)
R = np.abs(np.sum((np.array(points) - mean_of_points) ** 2, axis = -1))
new_points = [None for _ in range(len(points))]
for i in range(len(points)):
    new_points[i] = *points[i], R[i]


centroids = np.zeros(shape=(k, 3))
for i in range(k):
    centroids[i, 0] =  np.random.uniform(min_X, max_X)
    centroids[i, 1] =  np.random.uniform(min_Y, max_Y)
    centroids[i, 2] =  np.random.uniform(np.min(R), np.max(R))
centroids

error = float('inf')
assignemnts = dict.fromkeys(list(range(k)))

while(error > 1e-8):
    new_centroids, assignemnts = k_mean_step(centroids=centroids.copy(), points=new_points, k=k)
    error = np.sum(np.abs(centroids - new_centroids))
    centroids = new_centroids



area = 10
fig = plt.figure()
for i in range(k):
    X = np.array(assignemnts[i])[:, 0]
    Y = np.array(assignemnts[i])[:, 1]
    plt.scatter(X, Y,
                s=np.ones(shape=(X.shape[0],)) * area,
                c=cmap(i),
                alpha=0.5,
                label=f'Segment {i}')

plt.scatter(centroids[:,0], centroids[:, 1], s=30, label='Centorids')
plt.legend()
plt.savefig('res04.jpg')