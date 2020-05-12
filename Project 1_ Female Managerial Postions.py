import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import contextily as ctx
from matplotlib import style
import matplotlib.pyplot as plt
import plotly.offline as pyo

#column order : Municipality, Male, Female
#read specific sheet of xlsx documenet, and specify columns
data = pd.read_excel('P9115 - Non-Financial Census of Municipalities, 2018.xlsx', sheet_name='Table 1.2', usecols=[1,3,5])
#read in municipality population data
pop_data = pd.read_csv('zaf_adm3_pop.csv')

pop_data = pop_data[['ADM3_EN','p20_24', 'p25_29',	'p30_34',	'p35_39',	'p40_44',	'p45_49','p50_54', 'p55_59', 'p_Total']]
# pop_data['Employable'] = pop_data.iloc[:, 1:9].sum(axis=1)
# pop_data = pop_data[['ADM3_EN', 'Employable']]
pop_data = pop_data[['ADM3_EN']]
pop_data.rename(columns={'ADM3_EN':'Municipality'}, inplace=True, errors='raise')

#store Total Province values in a separate dataframe for later use
provinces_list = ['Western Cape', ' Eastern Cape', 'Northern Cape', 'Free State', 'KwaZulu-Natal', ' North West', 'Gauteng', ' Mpumalanga', 'Limpopo']
provinces_df = data[data['Unnamed: 1']=='TOTAL']

'''
the columns showing gender and Province names subheadingsexported  NaN 
only interested in the municipality and the gender values not heading names.
Drop NaN rows, not columns
'''
#this code removes TOTAL rows
data.dropna(axis=0, how='any', inplace=True)
data.reset_index(drop=True, inplace=True)

#remove district municipalities and total South Africa row
for i in data['Unnamed: 1']:
    if 'District Municipality' in i:
        #data.drop(data.loc[data['Unnamed: 0']==i].index, inplace=True)
        #or
        data = data[data['Unnamed: 1'] != i]
    
    elif 'SOUTH AFRICA' in i:
        data = data[data['Unnamed: 1'] != i]


data.drop(data.loc[data['Unnamed: 1']=='TOTAL'].index, inplace=True)

#remove part of string 'Local/Metropolitan Municipality'
for x in range(len(data['Unnamed: 1'])):
    #for  name in data['Unnamed: 0']:
        if 'Local Municipality' in data.iloc[ x, 0 ]:
            data.iloc[ x, 0 ] = data.iloc[ x, 0  ].replace('Local Municipality', '')
            data.iloc[ x, 0 ] = data.iloc[ x, 0 ].rstrip().lstrip() #remove excess white space
            
        
        elif 'Metropolitan Municipality' in data.iloc[ x, 0 ]:
            data.iloc[ x, 0 ] = data.iloc[ x, 0  ].replace('Metropolitan Municipality', '')
            data.iloc[ x, 0 ] = data.iloc[ x, 0 ].rstrip().lstrip() #remove excess white space
            

#rename columns   
data.rename(columns={'Unnamed: 1':'Municipality', 'Unnamed: 3':'Male Positions', 'Unnamed: 5':'Female Positions'},inplace=True, errors='raise')
# print(data.info())
# print(data.head())
data.replace('Madibeng', 'Local Municipality of Madibeng', inplace=True)
name_mismatch_check = []
# for name in data['Municipality']:
#     if name not in pop_data['Municipality'].tolist(): #has to be a list
#         name_mismatch_check.append(name)
# print('in gender Data but not in population:')
# print(name_mismatch_check)
# name_mismatch_check2 = []
# for name in pop_data['Municipality']:
#     if name not in data['Municipality'].tolist(): #has to be a list
#         name_mismatch_check2.append(name)
# print('in population but not in gender Data:')
# print(name_mismatch_check2)

data = data.merge(pop_data, on='Municipality')
data['Tot_Pos'] = data['Male Positions'] + data['Female Positions'] 
data['Diff'] = data['Male Positions'] - data['Female Positions'] 

#divide difference by total positions to get rid of population size bias
'''
100 male vs 80 female should be the same as 10 male vs 8 female in another dictrict
if we dont divde by total postins avalaible it'll seem ass if the 100 v 80 district
is more male dominant when it's not...

'''
data['M-F%'] = data['Diff']/data['Tot_Pos']
#scaling algorithm
for y in range(len(data['M-F%'])):   
    if (data.iloc[y,5] == 1)|(data.iloc[y,5] == -1):
        data.iloc[y,5] = data.iloc[y,4] / 15
#NaN are now where total was 0    

#print(data.info())

#compare province name in each df
#no need to sort alphabetically , merger tAKES care of that?
#search for missing data

################################################MAP CODE##########################################################################

# # Set the dimension of the figure
# my_dpi=96
# plt.figure(figsize=(2600/my_dpi, 1800/my_dpi), dpi=my_dpi)


# choose a preferred style
mpl.style.use('seaborn-dark')  
#print(plt.style.available) # see what styles are available
mapdata = gpd.read_file('zaf_admbnda_adm3_2016SADB_OCHA.shp')
mapdata = mapdata[['ADM3_EN','geometry']]
mapdata.rename(columns={'ADM3_EN':'Municipality'}, inplace=True)
#Web map tiles are typically provided in Web Mercator (EPSG 3857), 
mapdata = mapdata.to_crs(epsg=3857)
data.replace('KhΓi-Ma', 'Khâi-Ma', inplace=True)

#print(mapdata.info())

# name_mismatch_check3 = []
# name_mismatch_check4 = []

# for name in data['Municipality']:
#     if name not in mapdata['Municipality'].tolist():
#         name_mismatch_check3.append(name)
# print('In data but not in map')
# print(name_mismatch_check3)

# for name in mapdata['Municipality']:
#     if name not in data['Municipality'].tolist():
#         name_mismatch_check4.append(name)
# print('In map  but not in data')
# print(name_mismatch_check4)

# print(data.info())
# print(mapdata.info())
#data.drop(33, inplace=True) #droppng duplicate row
# duplicateRowsDF = mapdata[mapdata.duplicated(['Municipality'])]
# print(duplicateRowsDF)
# duplicateRowsDF2 = data[data.duplicated(['Municipality'])]
# print(duplicateRowsDF2)
mapdata = mapdata.merge(data, on='Municipality', how='left') #use only column from map
mapdata['M-F%'].fillna((mapdata['M-F%'].mean()), inplace=True)
mapdatashow = mapdata[(mapdata['M-F%']<0) | (mapdata['M-F%']>0.9)]
#merged = pd.merge(mapdata, data, on = 'Municipality', how='left')
'''
Scheme must be in the set: dict_keys(['boxplot', 'equalinterval', 'fisherjenks', 
'fisherjenkssampled', 'headtailbreaks', 'jenkscaspall', 'jenkscaspallforced', 
'jenkscaspallsampled', 'maxp', 'maximumbreaks', 'naturalbreaks', 'quantiles', 
'percentiles', 'stdmean', 'userdefined'])
'''
# schemes = ['boxplot', 'equalinterval', 'fisherjenks', 
# 'fisherjenkssampled', 'headtailbreaks', 'jenkscaspall', 'jenkscaspallforced', 
# 'jenkscaspallsampled', 'maxp', 'maximumbreaks', 'naturalbreaks', 'quantiles', 
# 'percentiles', 'stdmean', 'userdefined']
#cmap = mpl.colors.ListedColormap(['#EBA254', '#673B68', '#563A64', '#563A64', '#43345C', '#43345C'])
#norm = mpl.colors.BoundaryNorm([-0.6,-0.1,0,0.1,0.4,0.5,1], cmap.N)
#ax = mapdata.plot(column='M-F%', cmap=cmap, scheme='boxplot', alpha=1, linewidth=0.4, edgecolor='#8C4676', legend=True)
cmap = mpl.colors.ListedColormap(['#c44513','#d8541c','#E66E37','#dd6e2e', '#e08045', '#673B68', '#563A64', '#563A64', '#43345C', '#2d1342', '#260e3a'])
norm = mpl.colors.BoundaryNorm([-1,-0.95,-0.5,-0.37,-0.2,0,0.2,0.37,0.5,0.95,1], cmap.N)

###################################################LABEL REGIONS OF INTEREST###################################################
mapdatashow['coords'] = mapdata['geometry'].apply(lambda x: x.representative_point().coords[:])

mapdatashow['coords'] = [coords[0] for coords in mapdatashow['coords']]

fig, ax = plt.subplots(1, 1)
mapdata.plot(column='M-F%', ax=ax, cmap=cmap, norm=norm, alpha=1, linewidth=0.5, edgecolor='#8C4676', legend=True, legend_kwds={'label': "Male vs Female Positions", 'orientation': "horizontal"})
#cbar.ax.tick_params(labelsize=5) 
# ax = mapdata.plot(column='M-F%', cmap=cmap, norm=norm, alpha=1, linewidth=0.3, edgecolor='#8C4676', legend=True)
# ax.set_facecolor('#E5E5E5')
# for idx, row in mapdatashow.iterrows():
#     ax.annotate(row['Municipality'], xy=row['coords'],horizontalalignment='center',verticalalignment='top', color='#161616', fontsize=8)
    
# for x,y in mapdatashow['coords']:
#     ax.text(x, y, row['Municipality'], color='white', fontsize=10)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_facecolor('#E5E5E5')
fig = plt.gcf()
fig.set_size_inches(15, 10, forward=True)
fig.patch.set_facecolor('#E5E5E5')
#plt.figure(figsize=(14, 20), facecolor=('#E5E5E5'))
#fig.savefig('CHOROPLETH_MAP_HD.png', dpi=100)
plt.show()
#print(mapdatashow.head())