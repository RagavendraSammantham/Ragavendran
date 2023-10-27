# Ragavendran
# earthquake prediction model using python

DATA SOURCE:
import numpy as np import pandas as pd
import matplotlib.pyplot as plt

import os print(os.listdir("../input"))
FEATURE EXPLORATION:


Index(['Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Depth Error', 'Depth Seismic Stations', 'Magnitude', 'Magnitude Type',
'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 'Horizontal Error', 'Root Mean Square', 'ID', 'Source', 'Location Source', 'Magnitude Source', 'Status'], dtype='object')
Figure out the main features from earthquake data and create a object of that features, namely, Date, Time, Latitude, Longitude, Depth, Magnitude.


data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']] data.head()
VISUALIZATION:
from mpl_toolkits.basemap import Basemap

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=- 180,urcrnrlon=180,lat_ts=20,resolution='c')

longitudes = data["Longitude"].tolist() latitudes = data["Latitude"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
#resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.) x,y = m(longitudes,latitudes)
In [9]:
fig = plt.figure(figsize=(12,10)) plt.title("All affected areas")
m.plot(x, y, "o", markersize = 2, color = 'blue') m.drawcoastlines() m.fillcontinents(color='coral',lake_color='aqua') m.drawmapboundary()
m.drawcountries() plt.show()

TRAINING AND EVALUATION


# demonstrate that the train-test split procedure is repeatable from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split # create dataset
X, y = make_blobs(n_samples=100) # split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) # summarize first 5 rows
print(X_train[:5, :])
# split again, and we should see the same split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) # summarize first 5 rows
print(X_train[:5, :])



HYPERPARAMETER TUNING

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model, verbose=0)

# param_grid = {
#     "neurons": [16, 64], 
#     "batch_size": [10, 20], 
#     "epochs": [10],
#     "activation": ['sigmoid', 'relu'],
#     "optimizer": ['SGD', 'Adadelta'],
#     "loss": ['squared_hinge']
# }
param_grid = {
    "neurons": [16], 
    "batch_size": [10, 20], 
    "epochs": [10],
    "activation": ['sigmoid', 'relu'],
    "optimizer": ['SGD', 'Adadelta'],
    "loss": ['squared_hinge']
}

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

best_params = grid_result.best_params_
best_params

2023-02-12 14:30:16.688729: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best performance.
2023-02-12 14:30:16.721324: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best performance.
2023-02-12 14:30:16.733601: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best performance.
2023-02-12 14:30:16.761165: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best performance.
2023-02-12 14:30:17.151828: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2023-02-12 14:30:17.151828: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2023-02-12 14:30:17.151827: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2023-02-12 14:30:17.164576: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2023-02-12 14:34:25.381389: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2023-02-12 14:34:25.461923: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)



{'activation': 'sigmoid',
 'batch_size': 20,
 'epochs': 10,
 'loss': 'squared_hinge',
 'neurons': 16,
 'optimizer': 'SGD'}

FEATURE ENGINEERING

data['Month'] = data['Date'].apply(lambda x: x[0:2])
data['Year'] = data['Date'].apply(lambda x: x[-4:])

data = data.drop('Date', axis=1)


data['Month'] = data['Month'].astype(np.int)


data[data['Year'].str.contains('Z')]
invalid_year_indices = data[data['Year'].str.contains('Z')].index

data = data.drop(invalid_year_indices, axis=0).reset_index(drop=True)


data['Year'] = data['Year'].astype(np.int)


data['Hour'] = data['Time'].apply(lambda x: np.int(x[0:2]))

data = data.drop('Time', axis=1)

data['Status'].unique()


array(['Automatic', 'Reviewed'], dtype=object)

data['Status'] = data['Status'].apply(lambda x: 1 if x == 'Reviewed' else 0)


import pandas as pd import numpy as np
import matplotlib.pyplot as plt import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv("data/train_values.csv")




Index(['building_id', 'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'has_superstructure_adobe_mud',
'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other', 'legal_ownership_status', 'count_families', 'has_secondary_use', 'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police', 'has_secondary_use_other'], dtype='object')
labels = pd.read_csv("data/train_labels.csv")
df = df.sort_values("building_id")
labels = labels.sort_values("building_id")
df.head()
df = df.sort_values("building_id")
labels = labels.sort_values("building_id")
df.head()
'damage_grade' correlation_matrix = df.corr()
def plot_deposit_correlations(data): '''
Isolates the deposit columns of the correlation matrix and visualize it. '''
deposit_correlation_column = pd.DataFrame(correlation_matrix[DEPOSIT_COLUMN].drop(DEPOSIT_COLU MN))
deposit_correlation_column = deposit_correlation_column.sort_values(by=DEPOSIT_COLUMN, ascending=False)
sns.heatmap(deposit_correlation_column) plt.title('Heatmap of Deposit Variable Correlations')
 plot_deposit_correlations(df) 
 

 categorical_vars = ["land_surface_condition", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status"] for var in categorical_vars:
df = fuck_naman(df, var)

df.to_csv('data/labeled_train.csv') df["damage_grade"]

2	148259
3	87218
1	25124
Name: damage_grade, dtype: int64
# Pie chart
labels = ['Damage 1', 'Damage 2', 'Damage 3']
sizes = [25124, 148259, 87218,]
# only "explode" the 2nd slice (i.e. 'Hogs') explode = (0, 0, 0)
fig1, ax1 = plt.subplots()
patches, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
for text in texts: text.set_color('white') text.set_size(13)
for autotext in autotexts: autotext.set_color('white') autotext.set_size(13)
#draw circle
centre_circle = plt.Circle((0,0),0.80,fc='black') fig = plt.gcf() fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle ax1.axis('equal')
plt.tight_layout() plt.show()
value_counts()




normalized_df=(df-df.min())/(df.max()-df.min()) df = normalized_df

X = df.drop("damage_grade", axis=1) y = df["damage_grade"]
targets = df["damage_grade"].unique() print(targets)
pca = PCA(n_components=2) X_r = pca.fit(X).transform(X) print(X_r.shape)
PCA_Df = pd.DataFrame(data = X_r
, columns = ['principal component 1', 'principal component 2']) print(PCA_Df.head())
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors): indicesToKeep = df['damage_grade'] == target
plt.scatter(PCA_Df.loc[indicesToKeep, 'principal component 1']
, PCA_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50) plt.legend((targets + .5) * 2)
plt.title('PCA of Damage dataset')
plt.xlabel("First Principal Component") plt.ylabel('Second Principal Component')
[0.5 1. 0. ]
(260601, 2)


X = df.drop("damage_grade", axis=1) y = df["damage_grade"]
targets = df["damage_grade"].unique() print(targets)
pca = PCA(n_components=3) X_r = pca.fit(X).transform(X) print(X_r.shape)

 PCA_Df = pd.DataFrame(data = X_r[0:2000]
, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
colors = ['r', 'g', 'b'] fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') for target, color in zip(targets,colors):
indicesToKeep = df['damage_grade'] == target ax.scatter(PCA_Df.loc[indicesToKeep, 'principal component 1']
, PCA_Df.loc[indicesToKeep, 'principal component 2'], PCA_Df.loc[indicesToKeep, 'principal component 3'], c = color, s = 50) plt.legend((targets + .5) * 2)
plt.title('3-Component PCA of Damage ') plt.xlabel("First Principal Component") plt.ylabel('Second Principal Component')
[0.5 1. 0. ]
(260601, 3)
Text(0.5, 0, 'Second Principal Component')


X = df.drop("damage_grade", axis=1).astype('int') y = ((df["damage_grade"] + 0.5) * 2).astype('int') lda = LDA(n_components=2)
dmg_lda = lda.fit_transform(X, y) print(dmg_lda)
l_x = dmg_lda[:,0] l_y = dmg_lda[:,1]
cdict={1:'red',2:'green',3:'blue'}
labl={1:'Class1',2:'Class2',3:'Class3'} for l in np.unique(y):
ix=np.where(y==l)
ax = plt.scatter(l_x[ix],l_y[ix],c=cdict[l],s=40, label=labl[l])
plt.title("LDA Analysis") plt.legend()

/home/jordanrodrigues/anaconda3/lib/python3.7/site- packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
warnings.warn("Variables are collinear.")
labels = ["MaxAbsScaler RandomForest", "MaxAbsScaler SGD", "StandardScaler ExtremeTrees", "MaxAbsScaler BernoulliNaiveBayes", "MaxAbsScaler ExtremeTrees"]

values = [.3720, .4521, .2893, .4397, .3116]

plt.bar(labels, values, color='tab:cyan') plt.xticks(rotation=70) plt.rcParams.update({'font.size': 15}) plt.title("AutoML Performance") plt.ylabel("F1 Score")


Text(0, 0.5, 'F1 Score')
data_new = data.head()    # I only take 5 rows into new data
data_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted
ndex is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row

data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col






