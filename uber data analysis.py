#!/usr/bin/env python
# coding: utf-8

# In[303]:


## Working with 4 millions of Data of Uber!!


# In[1]:


pwd


# In[2]:


cd C:\Users\shaqu\Desktop\Data Science Projects\uber data analysis


# In[107]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


# In[6]:


df= pd.read_csv('uber-raw-data-apr14.csv')


# In[7]:


df


# lat ~ The latitude of the Uber pickup
# 
# 
# lon ~ The longitude of the Uber pickup
# 
# 
# base ~  The TLC base company code affiliated with the Uber pickup

# In[8]:


df['Date/Time']


# In[18]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[14]:


dt= '4/1/2014 0:11:00'
dt


# In[19]:


d, t= dt.split(" ")


# In[28]:


print(d)
print(t)


# In[23]:


d.split('/')


# In[29]:


t.split(':')


# In[30]:


m, d, y= d.split('/')


# In[31]:


print(m)
print(d)
print(y)


# ###  another way is to use datetime function!!

# In[33]:


dt


# In[34]:


dt= df['Date/Time']


# In[35]:


dt


# In[36]:


dt= pd.to_datetime(dt)


# In[39]:


dt


# In[44]:


# dt.months

# #error AttributeError: 'Series' object has no attribute 'months'


# In[45]:


dt= '2014-04-01 00:11:00'


# In[46]:


dt= pd.to_datetime(dt)


# In[47]:


dt


# In[49]:


dt.day


# In[50]:


dt.month


# In[51]:


dt.year


# In[52]:


dt.time


# In[53]:


dt.minute


# In[54]:


dt.hour


# In[ ]:


## convert datetime and add some useful columns!!


# In[334]:


df['Date/Time']= df['Date/Time'].map(pd.to_datetime)


# In[335]:


df['Date/Time']


# In[341]:


dt= df['Date/Time'][10]


# In[342]:


dt.day


# In[343]:


def get_dateOfMonth(dt):
    return dt.month
df['dateOfMonth']= df['Date/Time'].map(get_dateOfMonth)


# In[83]:


def get_dateOfDay(dt):
    return dt.day
df['dateOfDay']= df['Date/Time'].map(get_dateOfDay)


# In[84]:


def get_dateOfYear(dt):
    return dt.year
df['dateOfYear']= df['Date/Time'].map(get_dateOfYear)


# In[86]:


def get_dateOfHour(dt):
    return dt.hour
df['dateOfHour']= df['Date/Time'].map(get_dateOfHour)


# In[88]:


def get_dateOfMinute(dt):
    return dt.minute
df['dateOfMinute']= df['Date/Time'].map(get_dateOfMinute)


# In[81]:


df.drop('DOM', inplace=True, axis=1)


# In[89]:


df


# In[92]:


plt.scatter(df.Lat, df.dateOfDay)
plt.show()


# In[93]:


plt.scatter(df.Base, df.dateOfHour)
plt.show()


# In[94]:


plt.scatter(df.Lat, df.Lon)
plt.show()


# In[98]:


plt.hist(df.dateOfMinute.values)


# In[100]:


(df.dateOfMinute).values


# In[120]:


dom_values


# In[121]:


dom_values.plot()


# In[136]:


plt.hist(dom_values, bins= 30, rwidth= .8, range=(0.5, 31.5))


# In[138]:


hist(dom_values, bins= 30, rwidth= .8, range=(0.5, 30.5))
xlabel('date of the month')
ylabel('frequency')
title('frequency of the uber car in apri; 2014 year')


# In[142]:


for i, rows in df.groupby('dateOfMonth'):
    print(i, len(rows))


# In[145]:


df.head()


# ## how to merge the different months datasets together!!

# In[291]:


cd C:\Users\shaqu\Desktop\Data Science Projects\uber data analysis


# In[297]:


import pandas as pd


files =[file for file in os.listdir('./uber data of months')]
all_months_data = pd.DataFrame()

for file in files:
    df= pd.read_csv('./uber data of months/' + file)
    all_months_data = pd.concat([all_months_data, df])
    all_months_data.to_csv('all_uber_data1.csv', index=False)


# In[301]:


all_months_data   ## this can't be load fully in excel!!


# In[305]:


## alert !!! it takes lots of time to operate


# In[302]:


# all_months_data['Date/Time']= all_months_data['Date/Time'].map(pd.to_datetime)


# In[379]:


all_months_data.reset_index(drop = True, inplace = True)


# In[380]:


all_months_data


# In[357]:


all_months_data['Date/Time']


# In[358]:


all_months_data.describe()


# So from here we are going to work on around 4millions of uber datasets!!
# 
# 
# be excited!!

# In[359]:


all_months_data.info()


# In[360]:


all_months_data.isnull().sum()


# In[ ]:





# In[ ]:





# In[392]:


def get_dateOfMonth(dt1):
    return dt1.month
all_months_data['Month']= all_months_data['Date/Time'].map(get_dateOfMonth)


# In[390]:


def get_dayOfMonth(dt1):
    return dt1.day
all_months_data['DayOfMonth']= all_months_data['Date/Time'].map(get_dayOfMonth)


# In[388]:


def get_minute(dt1):
    return dt1.minute
all_months_data['minute']= all_months_data['Date/Time'].map(get_minute)


# In[431]:


def get_hour(dt1):
    return dt1.hour
all_months_data['Hour']= all_months_data['Date/Time'].map(get_hour)


# In[517]:


def get_weekday(dt1):
    return dt1.weekday()
all_months_data['WeekDaydff']= all_months_data['Date/Time'].map(get_weekday)


# In[521]:


# all_months_data.drop(columns= 'WeekDaydff', inplace = True)
all_months_data


# In[396]:


# all_months_data.drop(columns= 'DateOfMonth', inplace = True)


# In[345]:


all_months_data['Date/Time']


# In[397]:


dtx= all_months_data['Date/Time'][1]


# In[398]:


dtx


# In[399]:


plt.scatter(all_months_data.Lat, all_months_data.Lon)
plt.show()


# In[401]:


plt.hist(all_months_data.minute.values)


# #### How to create a groupby function from the counting the lenth of the dataset and then use it as groupby function!!

# In[402]:


def count_rows(rows):
    return len(rows)


# In[404]:


count_rows(all_months_data)


# In[417]:


def count_columns(columns):
    return all_months_data.shape[1]


# In[418]:


count_columns(all_months_data)


# In[422]:


by_date= all_months_data.groupby('Month').apply(count_rows)
by_date


# In[423]:


by_date.plot()
plt.show()


# in the month of September we got most data around 1 million!!

# In[ ]:


by_day= all_months_data.groupby('DayOfMonth').apply(count_rows)
by_day.head()


# In[427]:


by_day.plot()
plt.show()


# most of the uber was taken at the last day of the month, around ~ 150000!!

# In[428]:


all_months_data.columns


# In[468]:


by_min= all_months_data.groupby('minute').apply(count_rows)
by_min.head()


# In[467]:


by_hour= all_months_data.groupby('Hour').apply(count_rows)
by_hour.head()


# In[434]:


by_hour.plot()
plt.show()


# most of the uber cabs was booked during the evening from around 3pm to 18pm.

# In[430]:


by_min.plot()
plt.show()


# At 3:10 pm the hike is so high, so uber driver should be very ready the fill the demand of that time !!

# In[466]:


sorted_min= by_min.sort_values()
sorted_min.head()


# In[447]:


sorted_min.plot()


# In[458]:


hr1= all_months_data.Hour


# In[459]:


hr1.hist(bins= 24)


# most of the cars booked at evening around 15pm to 21 pmm!!

# In[462]:


day1= all_months_data.DayOfMonth


# In[465]:


day1.hist(bins= 31)


# In[471]:


lat_lon= all_months_data['Lat'].groupby(all_months_data['Lon'])


# In[481]:


lat_lon.describe()


# In[495]:


all_months_data['WeekDay'].unique.v


# In[500]:


len(pd.unique(all_months_data['WeekDay']))


# In[505]:


all_months_data['WeekDay'].nunique(axis=1)


# In[522]:


# Dataframe.col_name.nunique()
all_months_data.WeekDay


# In[545]:


hist(all_months_data.WeekDay, bins= 7, range=(-.5, 6.5), rwidth= .8, color= 'red', alpha= 0.4)
xlabel('weekdays')
ylabel('frequencies')
xticks(range(7), 'MOn, Tue, Wed, Thurs, Fri, Sat, Sun'. split())
plt.show()


# so from our graph we cann see that on wednesday and thursday highest numbers of cabs booked!!

# In[ ]:


# bar(all_months_data.WeekDay, height= 10, width=6)
# xlabel('weekdays')
# ylabel('frequencies')
# plt.show()


# In[550]:


dx_wd= all_months_data.WeekDay
pie_labels= ['MOn, Tue, Wed, Thurs, Fri, Sat, Sun']


# In[ ]:


# plt.pie(dx_wd )  pie chart is taking lot of time time to execute
get_ipython().getoutput('')


# ## cross analysing of oour datasets!!

# In[555]:


dx


# group by function to create a hour and week day of the whole datasets!!

# In[575]:


dx_wh= dx.groupby('WeekDay Hour'.split()).apply(count_rows).unstack()  ## This is amazing!!


# how to rename thbe index values in pandas?

# In[576]:


dx_wh.rename(index={'0': 'Monday', '1': 'Tuesday'})


# In[590]:


dx_wh.index.rename('0','MOn')  # Not find the right way to chamge the index names !!


# In[607]:



dx_wh


# In[612]:


sns.heatmap(dx_wh)
xlabel('hours')
ylabel('WeekDays')


# Analysing the Latitudes and longitudes!!

# In[621]:


hist(dx.Lat, bins=100,  range= (40.5, 41))
plt.show()


# In[630]:


hist(dx.Lon, bins=100,  range= (-74.1, -73.9))
plt.show()


# In[637]:


hist(dx.Lon, bins= 105, color= 'green')
plt.show()


# ATTACHING TWOO GRAPHS TOGETHER!!

# In[659]:


hist(dx.Lon, bins= 100, range=(-74.1, -73.9), color= 'green', label= 'Longitude')
legend(loc='upper right')
twiny()
hist(dx.Lat, bins=100,  range= (40.5, 41.5), color= 'red', label= 'Latitude')
legend(loc= 'upper left')

plt.show()


# In[663]:


plot(dx.Lon )
xlim(0,100000)
plt.show()


# In[664]:


plot(dx.Lat )
xlim(0,100000)
plt.show()


# In[665]:


plot(dx['Lat Lon'.split()])


# In[668]:


plot(dx['Lat'], dx['Lon'])


# In[669]:


# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000  for increasing the chunl size of the plot!!


# In[670]:


plot(dx['Lat'], dx['Lon'], '.')


# In[671]:


plot(dx['Lat'], dx['Lon'], '.', alpha = 0.4)


# So from our Latitudianl and longitudinal graph we can understand that at the longitude from -74,0 to -72.5 and latidude from 40.5 to 41 the max area covered by cab

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




