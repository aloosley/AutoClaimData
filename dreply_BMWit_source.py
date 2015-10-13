import pandas as pd
import re
#import matplotlib.pylab as plt # plotting
from geopy.geocoders import Nominatim # reverse geo-coding
import numpy as np
#import scipy as sp
from datetime import datetime
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
#from sklearn import cross_validation
#from sklearn.feature_selection import SelectKBest
import seaborn as sns
import requests
from IPython.core.display import HTML


# Get DF of data columns and datatypes:
def summarize(df):
    fieldInds = np.array([0,2,3,4,5,13,15,23,24,28,33,35,42,46,47,48,49,50,51,55,57,67])
    return df.dtypes[fieldInds]

def parse_and_load(path):
    # Parse (Replace pipe delimiters with commas):
    with open(path, 'r') as fp:
        data = re.sub(r'[|]', ',', fp.read())
    with open('claims_full_csv.csv', 'w') as fp:
        fp.write(data)
    
    # Load data into dataframe (similar to SQL):
    return pd.read_csv('claims_full_csv.csv')


def transform_vectorize(df):
    """
    This function does the following in order:
    1. Gets Region Definitions (by state)
    2. Makes all states lower case
    3. Converts dates to julian and defines v_age
    4. Creates a list of covariates
    5. Removes rows without Claim Age data
    7. Keeps only vehicles with at least 100 claims
    7a. Removes rows with negative claim values
    8. Creates Vehicle makes
    9.
    10. Removes 15 states (for reverse geo-code demo)
    """ 
    
    
    
    # 1. State to region mapper:
    state_to_reg = pd.read_excel('../data/USRegions.xlsx')
    state_to_reg.state = state_to_reg.state.map(lambda x: x.lower())
    state_to_reg_series = state_to_reg.set_index('state')['region_num']
    
    # 2. Make states lower case (for region mapping later):
    def statemap(state):
        if type(state) is not str:
            return state
        else:
            return state.lower()
    df.CLAIM_DEALER_STATEPROVINCE = df.CLAIM_DEALER_STATEPROVINCE.map(statemap)
    
    # 3. Convert dates into julian dates (days)
    df.PROD_DATE = pd.to_datetime(df.PROD_DATE).map(lambda x: x.to_julian_date())
    df.CLAIM_DATE = pd.to_datetime(df.CLAIM_DATE).map(lambda x: x.to_julian_date())
    df['v_age'] = df.CLAIM_DATE-df.PROD_DATE
    
    # 4. Create covariates list for models:
    covariates = []
    covariates.append('v_age')
    covariates.append('pCLAIM_AGE_BUCKET')
    
    # 5. Remove rows will nulls in pCLAIM_AGE_BUCKET col:
    df = df[df.pCLAIM_AGE_BUCKET.notnull()]
    
    
    # 7. Keep only models with more than 100 claims:
    claims = 100
    makes = df.groupby('Vehicle_Make').apply(lambda x: len(x)) > claims
    df = df[makes[df.Vehicle_Make].values]
    
    # 7a. Remove -ve claim values:
    df = df[df.pCLAIM_TOT_GLOBL_AMT>0]
    
    # 8. Vehicle_Makes Vector:
    makes = np.unique(df.Vehicle_Make)
    for idx,make in enumerate(makes):
        makeCol = 'm%02i' % (idx+1)
        df[makeCol] = (df.Vehicle_Make==make).astype(int)
        covariates.append(makeCol)
    
    # 9. Complaint:
    
    
    
    # 10. Randomly Remove 15 states:
    num_states_to_remove = 15
    df.ix[np.random.choice(df.index, size=num_states_to_remove, replace=False),
          'CLAIM_DEALER_STATEPROVINCE'] = np.nan
    
    # COMPLETE - return transformed data:
    return df, covariates, state_to_reg_series, makes
    
    
def lmplot(data, size=10, xlim=(0,None), ylim=(0,None)):
    sns.set(style="ticks", context="talk")

    # Make a custom sequential palette using the cubehelix system
    pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)
    
    # Plot tip as a function of toal bill across days
    g = sns.lmplot(x="v_age",
                   y="pCLAIM_TOT_GLOBL_AMT",
                   hue="Vehicle_Model",
                   data=data,
                   palette=pal, size=size)
    g.set(ylim=ylim)
    g.set(xlim=xlim)
    # Use more informative axis labels than are provided by default
    g.set_axis_labels("Vehicle Age (days)", "Claim Value ($)")
    
    
def reverseGEO_full(latlng):
    geolocator = Nominatim()
    
    # Fail if input is not a string:
    if type(latlng) is not str:
        return np.nan
        
    latlng = re.sub(r'[\s]', ', ', latlng)

    # Continue to try for reverse geolocation until successful:
    attempting = True
    attempt = 1
    while attempting and attempt < 5:
        try:
            location = geolocator.reverse(latlng, timeout=5)
            attempting = False # location acquisition complete
        except:
            print "attempt %02i failed... reattempting" % attempt
            attempt += 1
            
        if attempt == 5:
            print "reverse geolocator failed..."
    return location
    
def reverseGEO(latlng):
    location = reverseGEO_full(latlng)

    if 'address' in location.raw and 'state' in location.raw['address']:
        print '('+latlng+') --> '+location.raw['address']['state'].lower()
        return location.raw['address']['state'].lower()    
    else:
        return np.nan


def places_nearby(latlng):
    latlngList = latlng.split(' ')
    lat = float(latlngList[0])
    lng = float(latlngList[1])

    app_id = "VSMGAIrMAxwul3lz4DSl"
    app_code = "SSeAfRiFSj2_77Sdgpebcg"

    baseURL = "http://places.cit.api.here.com/places/v1/discover/explore"
    extURL = "?at=%f,%f&app_id=%s&app_code=%s&tf=plain&pretty=true" % (lat,lng,app_id,app_code)
    reqURL = baseURL+extURL
    
    return requests.get(reqURL).json()

def places_nearby_statement(nearby_places, closer_than=1000):
    count = 1
    print("DEALER ADDRESS: \n%s" % nearby_places['search']['context']['location']['address']['text'])
    print("")
    print("NEARBY LOCATIONS within %im:" % closer_than)
    for placeIdx in xrange(len(nearby_places['results']['items'])):
        if nearby_places['results']['items'][placeIdx]['distance'] < closer_than:
            print("%i. %im away, <here>, there is a %s (store)" % 
                  (count, nearby_places['results']['items'][placeIdx]['distance'],
                  nearby_places['results']['items'][placeIdx]['category']['id'])
                 )
            count += 1