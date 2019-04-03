__author__ = 'martinbodocky'

from pandas import Series, DataFrame
import pandas as pd
import numpy as np

obj = Series([4, -7, 5, 3])
obj.values
obj.index

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2[obj2 > 0]
obj2 ** 2

'b' in obj2
'e' in obj2

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)

pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()

obj5 = obj3 + obj4
obj5.name = 'population'
obj5.index.name = 'state'

obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']

#data frames
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)

DataFrame(data, columns=['year', 'state', 'pop'])

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])

frame2.ix['three']
frame2.year
frame2['state']

frame2['debt'] = 16.5
frame2['debt'] = np.arange(5.)
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val

frame2['eastern'] = frame2.state == 'Ohio'
del frame2['eastern']

# create data frame from dict of dict format
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3.T

DataFrame(pop, index=[2001, 2002, 2003])

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)

frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3.values

#Index Objects
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index[1:]

#index objects are immutable
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index

#reindexing
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj3 = obj.reindex(['a', 'b', 'r', 'k', 'e'], fill_value=0)

# reindexing methods
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')

frame = DataFrame(np.arange(9).reshape((3,3)), index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])

frame2 = frame.reindex(['a', 'b', 'c', 'd'])

states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)

#reindexing with filling
frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill',
              columns=states)
frame.ix[['a', 'b', 'c', 'd'], states]

#dropping entries from an axis
obj = Series(np.arange(5.), index = ['a','b','c','d','e'])
new_obj = obj.drop('c')

data = DataFrame(np.arange(16).reshape((4,4)),
                 index = ['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])

data.drop(['Colorado', 'Ohio'])
data.drop('two', axis=1)

#index, selection and filtering
obj = Series(np.arange(4.), index=['a','b','c','d'])
obj['b']
obj[1]
obj[['a','c']]
obj[obj>2]


data = DataFrame(np.arange(16).reshape((4,4)),
                 index = ['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data['two']
data[['three', 'one']]
data[data['three'] > 5]
data < 5
data.ix['Colorado', ['two', 'three']]
data[data<5]=0
data.ix[['Colorado', 'Utah'], [3,0,1]]
data.ix[:'Utah']
data.ix['Utah':]


#Arithemtic and data alignment
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])

#we add together only those which are the same
s1+s2

df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                index=['Utah', 'Ohio', 'Texas', 'Oregon'])

df1 + df2

#Arithemtic methods with fill values
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))

df1 + df2

#when value missing, we put 0 and add dataframes together
df1.add(df2, fill_value=0)

#operatitions between data frame and series
arr = np.arange(12.).reshape((3,4))
arr - arr[0]

frame =  DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas','Oregon'])

series = frame.ix[0]
# remove values for Utah
frame - series

#create a new series and add to data frame
series2 = Series(range(3), index=list('bef'))
frame + series2

series3 = frame['d']
frame.sub(series3, axis=0)

#function application and mapping

frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])

np.abs(frame)

f = lambda x : x.max() - x.min()
frame.apply(f)
frame.apply(f, axis=1)

def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)

format = lambda x: '%.2f' % x
frame.applymap(format)
frame['e'].map(format)

#Sorting and ranking
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()

frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])

frame.sort_index()
frame.sort_index(axis=1)

obj = Series([4, 7, -3, 2])
obj.order()

obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.order()

frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_index(by='b')

#ranking
obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()

#axis indexes with duplicate values
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique
obj['a']

df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])

#Summarizing and computing descriptive statistics
# all statistics methods consider to exclude missing values

df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
               index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])

df.sum()
df.sum(axis=1)
df.mean()
df.mean(axis=1)
df.mean(axis=1, skipna=False)

df.describe()

obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()

#correlation and covariance - yahoo web service is not working
# import pandas.io.data as web
#
# all_data = {}
# for ticket in ['AAPL', 'IBM', 'MSFT','GOOG']:
#     all_data[ticket] = web.get_data_yahoo(ticket, '1/1/2000', '1/1/2010')
#
# price = DataFrame({tic : data['Adj Close'] for tic, data in all_data.iteritems()})
# volume = DataFrame({tic :  data['Volume'] for tic, data in all_data.iteritems()})


#unique values, value count and membership
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])

uniques = obj.unique()
obj.value_counts()
pd.value_counts(obj.values, sort=False)
mask = obj.isin(['b', 'c'])
obj[mask]

#Handling Missing Data
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull()

from numpy import nan as NA
data = Series([1, NA, 3.5, NA, 7])

data.dropna()
data[data.notnull()]

data = DataFrame([[1., 6.5, 3.],
                  [1., NA, NA],
                  [NA, NA, NA],
                  [NA, 6.5, 3.]])

#drop each row which conrains NA
data.dropna()
#drop just row with all NA
data.dropna(how='all')
#dropping columns
data[4] = NA
data.dropna(axis=1, how='all')

#Filling in Missing Data
df = DataFrame(np.random.randn(7, 3))
df.ix[:4, 1]= NA; df.ix[:2,2]

df.fillna(0)
df.fillna({1:0.5, 3:-1})
#always returns a reference to the filled object
_ = df.fillna(0, inplace=True)

df = DataFrame(np.random.randn(6, 3))
df.ix[2:, 1] = NA; df.ix[4:, 2] = NA

df.fillna(method='ffill')
df.fillna(method='ffill', limit=2)

data = Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())

#Hierarchical Indexing
data = Series(np.random.randn(10),
              index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],[1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data.index
data['b']
data['b':'c']
data.ix[['b','d']]
data.unstack()
data.unstack().stack()

frame = DataFrame(np.arange(12).reshape((4,3)),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=[['Ohio', 'Ohio', 'Colorado'],
                           ['Green', 'Red', 'Green']])

frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame['Ohio']

#Reordering and sorting levels
frame.swaplevel('key1', 'key2')
#summary by level
frame.sum(level = 'key2')
frame.sum(level = 'color', axis=1)

#Using a dataframe's columns
frame = DataFrame({'a': range(7), 'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})

frame2 = frame.set_index(['c','d'])
frame.set_index(['c','d'], drop=False)
frame2.reset_index()