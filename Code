import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import sklearn
import csv 
import matplotlib.cm as cm  
from geopy.geocoders import Nominatim
from matplotlib.colors import ListedColormap
from tqdm import tqdm
tqdm.pandas()

#Input Global Landslide Catalog
df = pd.read_csv(r"D:\Code\Visual Studio Code\Global_Landslide_Catalog_Export_rows.csv")


#Printing First 5 Rows of the Dataset
pd.set_option("display.max_columns", None)
print(df.head())

#Understanding Data 
print("===DATASET INFO===")
print("Number of rows: ", len(df))
print("Columns Information: ")
print(df.info())

#Cleaning Data 

##Removing Duplicates Rows
### Removing Part 1 
dupe_rows1 = df[df.duplicated(subset=['event_date','country_name'], keep=False)]
df = df.drop_duplicates(subset=['event_date','country_name'], keep='first')
### Removing Part 2
dupe_rows2 = df[df.duplicated(subset=['event_title'], keep=False)]
df = df.drop_duplicates(subset=['event_title'], keep='first')

##Handling Missing Values
### 0 : Finding Out total of missing value in each column
print(df.isna().sum())

### 1 : Deleting columns with too much missing values (missing values > 0.5 => column removed)
missing_values = df.isnull().sum()
threshold = len(df) * 0.5
cols_to_drop = missing_values[missing_values > threshold].index
df = df.drop(columns=cols_to_drop)
print("Columns dropped: ", cols_to_drop)

# Columns dropped: Index(['event_time', 'injury_count', 'storm_name', 'photo_link', 'notes'], dtype='object')

### 2 : Dropping rows of data with missing values in selected columns
### Max amount of missing value in a column to still be considered for this operation is 100   
missing_counts = df.isnull().sum()
for col in missing_counts[missing_counts < 120].index:
    df = df[df[col].notnull()]
print("Number of rows after dropping rows with missing values in selected columns: ", len(df))

#Number of rows after dropping rows with missing values in selected columns:  7704

### 3: Filling missing event_time with time of event_date
df['event_date'] = df['event_date'].str.strip().str.replace('  ', ' ')
df['event_date'] = pd.to_datetime(df['event_date'],errors='coerce')
df['event_time'] = df['event_date'].dt.time
df['event_time'] = pd.to_datetime(df['event_time'],errors='coerce')


###4 : Filling country_code with code derived from latitude and longitude
geolocator = Nominatim(user_agent="geoapi")
def get_country_code(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en', exactly_one=True)
        if location and 'country_code' in location.raw['address']:
            return location.raw['address']['country_code'].upper()
        else:
            return 'unknown'
    except:
        return 'unknown'
mask = df['country_code'].isna() | (df['country_code'] == 'unknown')
df.loc[mask, 'country_code'] = df.loc[mask].progress_apply(
    lambda row: get_country_code(row['latitude'], row['longitude']),
    axis=1
)

###5 : Filling missing valued-cell in a column with a certain value
### Filling missing fatality_count with 0
#Removing km from location_accuracy column & converting exact to 0 km
print(df['location_accuracy'].value_counts())
df['location_accuracy'] = df['location_accuracy'].str.replace('km', '')
df['location_accuracy'] = df['location_accuracy'].replace('exact', '0')
df['fatality_count'] = df['fatality_count'].fillna(0)

###6 : Filling the rest of missing valued-cell in a column with 'unknown'
df = df.fillna('unknown')

###7 : Replacing 'unknown' with modus value of the column
df['location_accuracy'] = df['location_accuracy'].replace('unknown', df['location_accuracy'].mode()[0])
print(df['location_accuracy'].value_counts())

##Converting the rest of date-related-column to Datetime datatype
df['submitted_date']=pd.to_datetime(df['submitted_date'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
df['created_date']=pd.to_datetime(df['created_date'],  format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
df['last_edited_date']=pd.to_datetime(df['last_edited_date'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
print(df[['event_date', 'submitted_date', 'created_date', 'last_edited_date']].dtypes)

##Handling location_accuracy column

##Converting numeric-related-column to numeric datatype
numeric_columns = ['location_accuracy','fatality_count', 'gazeteer_distance','latitude', 'longitude']
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
print(df[numeric_columns].dtypes)  

##Removing outliers in numeric-related-column
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

columns_to_clean = ['location_accuracy']
for col in columns_to_clean:
    df = remove_outliers_iqr(df, col)


##Data Transformation or Encoding Categorical Data

##Converting event_time to 24-hour format
df['event_hour'] = df['event_date'].dt.hour

##Creating fatality category from fatality_count
bins = [-1, 0, 5, 20, 100, np.inf]
labels = ['None', 'Low (1–5)', 'Moderate (6–20)', 'High (21–100)', 'Extreme (>100)']
df['fatality_category'] = pd.cut(df['fatality_count'], bins=bins, labels=labels)

##Creating time category from event_hour
bins = [0, 8, 16, 23, np.inf]
labels = ['Day (0–8)', 'Morning (8–16)', 'Evening (16–23)', 'Night (>23)']
df['event_hour_category'] = pd.cut(df['event_hour'], bins=bins, labels=labels)

#Data Reduction : Feature Selection
##Removing columns with the same value in all rows
### Removing country_name || Contain same value as country_code
df = df.drop(columns=['country_name'])

##Removing detailed location information
df = df.drop(columns=['location_description'])

##Removing columns that are not useful for analysis
columns_to_remove = ['source_name','event_id', 'source_link', 'event_description', 'event_import_id', 'event_import_source', 'submitted_date', 'created_date', 'last_edited_date', 'admin_division_name', 'admin_division_population', 'event_title']
df.drop(columns=columns_to_remove, inplace=True)


df.to_csv("D:\Code\Visual Studio Code\cleaned_dataset.csv", index=False)

#Insight Gained
##Longitude and Latitude of Landslide Events
plt.scatter(df["longitude"], df["latitude"], alpha=0.5)
plt.show()

print(df.info())
##Heatmap of numerical data
selected_columns = df.select_dtypes(include=[np.number]).columns.tolist()
sns.heatmap(df[selected_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Heatmap of Numerical Data")
plt.show()

#Longitude and Latitude to different type of columns 
##To location_accuracy
df_sorted = df.sort_values(by='location_accuracy', ascending=False)
reversed_viridis = ListedColormap(cm.get_cmap('viridis').colors[::-1])
sns.scatterplot(
    data=df_sorted,
    x="longitude",
    y="latitude",
    hue="location_accuracy",  
    palette=reversed_viridis,        
    alpha=1.0
)
plt.title("Landslide Events Colored by Location Accuracy")
plt.show()


# To fatality_count presented in category form (fatality_category)
sns.scatterplot(
    data=df,
    x="longitude",
    y="latitude",
    hue="fatality_category",  
    palette="viridis",        
    alpha=0.6
)
plt.title("Landslide Events Colored by Fatality Count presented in Category Form (fatality_category)")
plt.show()

# To event_time presented in 'event_hour_category'
sns.scatterplot(
    data=df,
    x="longitude",
    y="latitude",
    hue="event_hour_category",  
    palette="viridis",        
    alpha=1.0
)
plt.title("Landslide Events Colored by Event Hour Category")
plt.show()

# To landslide_size
sns.scatterplot(
    data=df,
    x="longitude",
    y="latitude",
    hue="landslide_size",  
    palette="viridis",        
    alpha=1.0
)
plt.title("Landslide Events Colored by Landslide Size")
plt.show()

# To landslide_size
sns.scatterplot(
    data=df,
    x="longitude",
    y="latitude",
    hue="landslide_trigger",  
    palette="viridis",        
    alpha=1.0
)
plt.title("Landslide Events Colored by Landslide Trigger")
plt.show()


