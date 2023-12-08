import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import folium
from folium import Choropleth,Circle,Marker
import seaborn as sns
import re
import os
from sklearn.cluster import KMeans
import pulp
import itertools

st.title('VRP')


df1=pd.read_excel('./RouteDbData.xlsx',engine='openpyxl')
df2=pd.read_csv('./Central_dump.csv')

df3=pd.read_excel('./Bharthal_Routes.xlsx',engine='openpyxl')
x=['Route_no','Origin Point','DC 1','DC 2','DC 3','DC 4','DC 5','DC 6','DC 7','DC 8','Origin Point','Distance 1','Distance 2','Distance 3','Distance 4','Distance 5','Distance 6','Distance 7','Distance 8','Distance Back','Travel Time 1','Travel Time 2','Travel Time 3','Travel Time 4','Travel Time 5','Travel Time 6','Travel Time 7','Travel Time 8','Travel Time Back','Origin Dispatch Time','ETA DC 1','ETA DC 2','ETA DC 3','ETA DC 4','ETA DC 5','ETA DC 6','ETA DC 7','ETA DC 8','ETA Origin', 'Distance till Last DC', 'Last DC to MFC']
df3.columns=x
df3=df3.drop(0).reset_index()
df3.drop('index',axis=1,inplace=True)

DC=pd.DataFrame([j for x in df3.iloc[:,2:10].values for j in x if str(j) != 'nan'],columns=['DC'])
DC=DC.merge(df2[['center_name','property_lat','property_long','center_pincode']],left_on='DC',right_on='center_name')
DC.drop('DC',axis=1,inplace=True)
DC.drop_duplicates(inplace=True)
DC['lat_long']=list(zip(DC['property_lat'],DC['property_long']))
DC.loc[51]=['Bharthal',28.540562,77.05089,110077,[28.540562,77.05089]]
org_cluster_dict = dict(enumerate([list(x) for x in df3.iloc[:,2:10].values]))
new_d = {val:int(key) for key, lst in org_cluster_dict.items() for val in lst}
DC['org_cluster']=DC['center_name'].map(new_d)
DC.rename(columns={'property_lat':'Lattitude','property_long':'Longitude'},inplace=True)
DC= DC.apply(np.roll, shift=1)
st.subheader('Raw data')
st.subheader('Here we look at a cleaned data set of center name lat-long and pincode, the last variable org_cluster is a K-means clustered group for selcting clusters for Linear Programming.')
st.markdown('We have performed k means clusters to look at the difference between the original cluster and K-means labeled clusters ')


# create kmeans model/object
kmeans = KMeans(
    init="random",
    n_clusters=10,
    n_init=10,
    max_iter=300,
    random_state=42
)
features = DC[['Lattitude','Longitude']]
kmeans.fit(features)
labels = kmeans.labels_
DC['k_mean_cluster']= labels
DC.reset_index(drop=True,inplace=True)
# Folium plot
st.write(DC)


m_2 = folium.Map(location=[28.66, 77.15], tiles=" OpenStreetMap", zoom_start=10)


colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', \
          'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', \
          'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', \
          'black', 'lightgray', 'red', 'blue', 'green', 'purple', \
          'orange', 'darkred', 'lightred', 'beige', 'darkblue', \
          'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', \
          'lightgreen', 'gray', 'black', 'lightgray']
for idx, row in DC.iterrows():
    if row['center_name'] == 'Bharthal':
        folium.Circle(radius=1000, location=[row['Lattitude'], row['Longitude']], fill=False, color='red').add_to(m_2)
        Marker([row['Lattitude'], row['Longitude']],
               icon=folium.Icon(color='green',icon='home'),
               popup=row['center_name']).add_to(m_2)

    else:
        # colors=['red', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige','darkgreen']
        for i in range(0, 10):
            if row['k_mean_cluster'] == i:
                Marker([row['Lattitude'], row['Longitude']],
                       icon=folium.Icon(color=colors[i]),
                       popup=row['center_name']).add_to(m_2)

# Define a Streamlit custom component
class FoliumMapComponent:
    def __init__(self, folium_map):
        self.folium_map = folium_map

    def write(self):
        self.folium_map.save("map.html")
        st.components.v1.html(open("map.html").read(), width=800, height=600)

# Create an instance of your custom FoliumMapComponent
folium_component = FoliumMapComponent(m_2)



# Display the Folium map in Streamlit
st.subheader("Folium Map on Streamlit, \nthe depot has a ring to the icon: This is the start and end of a route, to differentiate the Icon is changed to :house: from ! ")
folium_component.write()


# TAT matrix
st.subheader("Let us look at the raw distance matrix used ")
dist= pd.read_csv('./MFC_DELH_Dis_tat - Distance.csv')
st.write(dist[:5])
st.markdown('We have the upper matrix to be null values and the distances are exracted from google distance API and the original format is as shown above for 5 records')
st.markdown('The transformed distance matrix and its cleaned format is:  ')

dist.rename(columns={'Unnamed: 0': 'org'},inplace=True)

dist.set_index('org',inplace=True)

for col in dist.columns:
    dist[col]= dist[col].apply(lambda x: float(x.split()[0]) if type(x)!= float else 0)

np.fill_diagonal(dist.values,0)

X= dist.values

X = X + X.T - np.diag(np.diag(X))

dist_MFC_1= pd.DataFrame(X,columns=dist.columns).set_index(dist.columns)
fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(dist_MFC_1, ax=ax, cmap='Blues', annot=True, fmt='.0f', cbar=True, cbar_kws={"shrink": .3}, linewidths=.1)
plt.title('distance matrix in KM')
# plt.show()
st.pyplot(fig)

st.write("Upload the demand file csv, Can download the static file displayed below and change the demand at centers")
#Uploading file -B
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

     # Can be used wherever a "file-like" object is accepted:
    demand_df = pd.read_csv(uploaded_file)
    st.write(demand_df)

    st.write(f'total current demand units are {len(demand_df)}')
else:
    demand_df = pd.read_csv('./demand_file_d1.csv')

st.write(demand_df)
st.write('Distribution of demand')
fig, ax = plt.subplots(figsize=(10,7))
ax.bar(demand_df['center_name'],demand_df['demand'],width=0.6)
ax.tick_params(axis='x', rotation=45,labelsize=7)

# show_every_nth_label = 2
# for i, label in enumerate(ax.get_xticklabels()):
#     if i % show_every_nth_label != 0:
#         label.set_visible(False)
ax.set_xlabel('Center Names', fontsize=12)
ax.set_ylabel('Demand', fontsize=12)
ax.set_title('Demand by Center', fontsize=14)
st.pyplot(fig)

# limit the cluster run
st.write(f'Due to limitations of computational power let us select only 2-3 clusters of the region')
#collect the list of cluster


numbers = st.text_input("PLease enter max 3 cluster numbers in range [0-9] comma separation allowed. If no input for demo will have the app run on clusters 4,5,6")
if numbers.strip():
    clusters = [int(i) for i in re.split("[^0-9]", numbers) if i != ""]
else:
    clusters = [4, 5, 6]

tat_df = pd.read_csv('./tat_matrix.csv')
tat_df=tat_df.set_index('Unnamed: 0')
tat_df.index.name=None

DC=DC.merge(demand_df,how='outer')

st.write(clusters)
study = DC[DC['org_cluster'].isin([np.nan]+clusters)]['center_name']
st.markdown('Based on cluster selection demand file is  ')

st.write(DC[DC['center_name'].isin(study)])
selection=list(DC[DC['center_name'].isin(study)&(DC['demand'].notna())]['center_name'])
tat=tat_df.loc[selection,selection]

st.markdown('Based on cluster selection and the non 0 demand centers, the filtered tat matrix is ')

st.write(tat)

# enter the vehicle capacity
vehicle_capacity = st.text_input("PLease enter vehicle capacity if none provided, will run it on 75")
if vehicle_capacity.strip():
    vehicle_capacity=int(vehicle_capacity)
else:
    vehicle_capacity = int(120)

demands = DC[DC['center_name'].isin(study)&(DC['demand'].notna())]['demand'].values

n_customer = len(tat) - 1
n_point = n_customer + 1


problem = pulp.LpProblem('cvrp_mip', pulp.LpMinimize)

# set variables
x = pulp.LpVariable.dicts('x', ((i, j) for i in range(n_point) for j in range(n_point)), lowBound=0, upBound=1,
                          cat='Binary')
n_vehicle = pulp.LpVariable('n_vehicle', lowBound=0, upBound=100, cat='Integer')

# set objective function
problem += pulp.lpSum([tat.iloc[:, :].values[i][j] * x[i, j] for i in range(n_point) for j in range(n_point)])

# set constrains
for i in range(n_point):
    problem += x[i, i] == 0

for i in range(1, n_point):
    problem += pulp.lpSum(x[j, i] for j in range(n_point)) == 1
    problem += pulp.lpSum(x[i, j] for j in range(n_point)) == 1

problem += pulp.lpSum(x[i, 0] for i in range(n_point)) == n_vehicle
problem += pulp.lpSum(x[0, i] for i in range(n_point)) == n_vehicle

# eliminate subtour
subtours = []
for length in range(2, n_point):
    subtours += itertools.combinations(range(1, n_point), length)

for subt in subtours:
    demand = np.sum([demands[s] for s in subt])
    arcs = [x[i, j] for i, j in itertools.permutations(subt, 2)]
    problem += pulp.lpSum(arcs) <= np.max([0, len(subt) - np.ceil(demand / vehicle_capacity)])
## time constraint
## solve proble
status = problem.solve()
# output status, value of objective function
# status, pulp.LpStatus[status], pulp.value(problem.objective), pulp.value(n_vehicle)
# output status, value of objective function
st.write(f" LP output is {status}, and LP status is {pulp.LpStatus[status]}")
st.write(f"objective function minimizing total distance is achiving the touchpoints in {pulp.value(problem.objective)} km, \n")
st.write(f"the number of vehicles required is {pulp.value(n_vehicle)}")

st.write('The Objective function defined in the LP problem is :')
objective_str = "Objective: "
for var in problem.variables():
    coefficient = var.varValue if var.varValue else 0  # Get variable value or default to 0
    objective_str += f"{coefficient} * {var.name} + "

# Remove the trailing " + " from the objective string
objective_str = objective_str[:-3]

# Display the objective in Streamlit
st.write(objective_str)

# st.write(problem.objective)
st.write(f'The no of constraints solving for the demands are {len(problem.constraints)}')
# st.write(len(problem.constraints))


study= DC[DC['center_name'].isin(selection)]
st.subheader('Mapping_cluster and the suggested route based on the demand and selected cluster run is ')

fig = plt.figure(figsize=(16, 16))

for i, row in DC.iterrows():
    if row['center_name'] == 'Bharthal':
        plt.scatter(row['Lattitude'], row['Longitude'], c='r')
        plt.text(row['Lattitude'] + 0.0002, row['Longitude'] + 0.0002, 'depot')
    else:  # row['org_cluster']==4:
        plt.scatter(row['Lattitude'], row['Longitude'], c='black')
        plt.text(row['Lattitude'] + 0.0002, row['Longitude'] + 0.0002, f'{i}')

# plt.xlim([study['Lattitude'].min()-0.1,study['Lattitude'].max()+0.1])
# plt.ylim([study['Longitude'].min()-0.1,study['Longitude'].max()+0.1])
plt.title('points: id')

# draw optimal route
cmap = plt.cm.get_cmap('Dark2')
routes = [(i, j) for i in range(n_point) for j in range(n_point) if pulp.value(x[i, j]) == 1]
print(pulp.value(n_vehicle))
vrp_routes = []
for v in range(int(pulp.value(n_vehicle))):
    # identify the route of each vehicle
    vehicle_route = [routes[v]]
    while vehicle_route[-1][1] != 0:
        for p in routes:
            if p[0] == vehicle_route[-1][1]:
                vehicle_route.append(p)
                break
    # for
    vrp_routes.append(vehicle_route)

    # draw for each vehicle
    arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor=cmap(v))
    for i, j in vehicle_route:
        plt.annotate('', xy=[study.iloc[j]['Lattitude'], study.iloc[j]['Longitude']],
                     xytext=[study.iloc[i]['Lattitude'], study.iloc[i]['Longitude']], arrowprops=arrowprops)

plt.show()
st.pyplot(fig)



