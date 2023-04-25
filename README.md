# HiSpot：High Intelligence Spatial Optimization 
Develop HiSpot open-source software to realize mathematical planning solver, approximate algorithm and heuristic algorithm to solve spatial optimization problems.

### Location Routing Problem
LRP jointly considers the facility location problem (FLP) and the vehicle routing problem (VRP)

```python
import random
import location
import numpy as np
import pandas as pd
import osmnx as ox

%% data process
df = pd.read_csv('../data/北京POI裁剪.csv',encoding='gbk')
df = df.query('adname=="东城区" | adname=="西城区" | adname=="朝阳区" | adname=="海淀区"').reset_index(drop=True)
data = df[['long', 'lat']]
num_points = df.shape[0]
num_located = 10
np.random.seed(0)
num_people = np.random.randint(1,2, size=num_points)
cartesian_prod = list(product(range(num_points), range(num_points)))
points = [(data['long'][i], data['lat'][i]) for i in range(num_points)]
facility_index = [1, 7, 27, 28, 29, 42, 45, 84, 105, 108]
all_idex=list(range(108))
set1 = set(facility_index)
set2 = set(all_idex)
demand_index = list(set1^set2)
real_num_fa = len(facility_index)
real_num_de = len(demand_index)
real_facility_nodes =[]
real_demand_nodes =[]
for i in facility_index:
    real_facility_nodes.append(points[i])
for j in demand_index:
    real_demand_nodes.append(points[j])
real_fa_cap = [(random.randint(50, 100),  random.randint(150, 200)) for i in range(real_num_fa)]
real_de_demand = [random.randint(5, 15) for i in range(real_num_de)]
real_fa_cost = [random.randint(0, 1) for i in range(real_num_fa)]
G = ox.load_graphml('../data/Beijing.graphml')

%% inference
real_selected_facility, real_unselected_facility, real_assigned = LRP_cap(facility_nodes=real_facility_nodes,
              demand_nodes=real_demand_nodes,
              fa_cap=real_fa_cap,
              de_demand=real_de_demand,
              solver=GUROBI_CMD()).prob_solve()
%% plot
ox.plot_graph(G, figsize=(50,50),bgcolor="#F5F5F5",node_size=0,edge_color = "#A4BE7B", show=False, close=False)
for j in range(num_points):
    if j in facility_index:
        if j in real_selected_facility:
            lx = df['lat'][j]
            ly = df['long'][j]
            plt.plot(ly, lx, c='red', marker='*', markersize=20,zorder=3)
        else:
            lx = df['lat'][j]
            ly = df['long'][j]
            plt.plot(ly, lx, c='#ff69E1', marker='*', markersize=20,zorder=2)
    else:
        lx = df['lat'][j]
        ly = df['long'][j]
        plt.plot(ly, lx, c="black",marker='o',markersize=20, zorder=1)
for i in range(len(real_assigned)):
    pts = [real_facility_nodes[real_assigned[i][0]], real_demand_nodes[real_assigned[i][1]]]
    plt.plot(*zip(*pts), c='Orange', linewidth=2, zorder=1)
plt.show()
```
see [notebook](https://github.com/HIGISX/HiSpot/blob/main/notebook/LRP-cap.ipynb) for more code details.

### Uncapacitated Facility Location Problem （UFLP）
import random
import numpy as np
import osmnx as ox
import pandas as pd

from location.FLPModel import *

```python
% data process
df = pd.read_csv('../data/北京POI裁剪.csv',encoding='gbk')
df = df.query('adname=="东城区" | adname=="西城区" | adname=="朝阳区" | adname=="海淀区"').reset_index(drop=True)
data = df[['long', 'lat']]
num_points = df.shape[0]
num_located = 10
np.random.seed(0)
num_people = np.random.randint(1,2, size=num_points)
demand = np.random.randint(20, size=num_points)  #d
cost = np.random.randint(20, size=num_points)  #c
cartesian_prod = list(product(range(num_points), range(num_points)))
points = [(data['long'][i], data['lat'][i]) for i in range(num_points)]

%% inference 
y, selected, selected_points, unselected_points  = UFLP(num_people=num_people,
                                                demand=demand,
                                                num_points=num_points,
                                                num_located=num_located,
                                                cartesian_prod=cartesian_prod,
                                                cost=cost,
                                                cover=points,
                                                solver=PULP_CBC_CMD()).prob_solve()

%% plot
G = ox.load_graphml('..\data\Beijing.graphml')
ox.plot_graph(G, figsize=(50,50),bgcolor="#F5F5F5",node_size=0,edge_color = "#A4BE7B", show=False, close=False)
for j in range(num_points):
    if j in selected:
        lx = df['lat'][j]
        ly = df['long'][j]
        plt.plot(ly,lx,c='red',marker='*',markersize=50, zorder=3)
    else:
        lx = df['lat'][j]
        ly = df['long'][j]
        plt.plot(ly,lx,c="black",marker='o',markersize=20, zorder=2)
#Lines
for i in range(num_points):
    for j in range(num_points):
        if y[i][j].varValue == 1 :
            pts = [points[i], points[j]]
            plt.plot(*zip(*pts), c='#1E90FF', linewidth=3.5, zorder=1)
```
see [notebook](https://github.com/HIGISX/HiSpot/blob/main/notebook/UFLP.ipynb) for plotting code and more details.


## Examples
- [p-Median](https://github.com/HIGISX/HiSpot/blob/main/notebook/pMedian.ipynb)
- [p-Center](https://github.com/HIGISX/HiSpot/blob/main/notebook/pCenter.ipynb)
- [P-Dispersion](https://github.com/HIGISX/HiSpot/blob/main/notebook/pDispersion.ipynb)
- [MCLP](https://github.com/HIGISX/HiSpot/blob/main/notebook/MCLP.ipynb)
- [LSCP](https://github.com/HIGISX/HiSpot/blob/main/notebook/LSCP.ipynb)
- [BCLP](https://github.com/HIGISX/HiSpot/blob/main/notebook/BCLP.ipynb)
- [MEXCLP](https://github.com/HIGISX/HiSpot/blob/main/notebook/MEXCLP.ipynb)
- ...


## Running Locally
1. Clone the repo `git clone https://github.com/HIGISX/HiSpot`
2. conda create -n higis python
3. conda activate higis
4. Launch jupyter notebook `jupyter notebook`

You should now be able to run the example notebooks.

You can choose to install and use another solver that is supported by [Pulp](https://github.com/coin-or/pulp):
- [GLPK](https://www.gnu.org/software/glpk/) (included in conda environment)
- [COIN-OR CBC](https://github.com/coin-or/Cbc)
- [CPLEX](https://www.ibm.com/analytics/cplex-optimizer)
- [Gurobi](https://www.gurobi.com/)

## Requirments
-numpy
-pulp

## Installation
pip install higis
pip install numpy
pip install pulp

## Contribute

## Support 

## Code of Conduct

## Citation 


