import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Problems I came accross:
#Date discrepencies between the two datasets. Since the Strava dataset uses UTC, some dates were rounded to the following day making multiple runs in a day.
#Meanwhile, the weather dataset uses local time (EST). The steps I took to resolve this are shown below.

#Loaded Strava Activities file (activities were only up to July 1st, 2025 as this was when I downloaded them)
df = pd.read_csv('C:\\Users\\Raymond\\Documents\\strava\\activities.csv')
#Loaded weather file, data comes from Open-Meteo free open source api
df2 = pd.read_csv('C:\\Users\\Raymond\\Downloads\\weatherData.csv', skiprows=3) #first three rows merely show coordinates and time zone selected

#Made sure all dates are also in DateTime objects. As mentioned above, entries in the Activities files are timestamped as UTC so I had to convert it to Eastern Standard Time, I'm not using the time but this is so the dates are shown correctly
df['Activity Timestamp'] = pd.to_datetime(df['Activity Date'], utc=True, errors='coerce')
df['Activity Local Time'] = df['Activity Timestamp'].dt.tz_convert("America/New_York")
df['Activity Date'] = df['Activity Local Time'].dt.normalize()
df['Activity Date'] = df['Activity Date'].dt.tz_localize(None)

df2['time'] = pd.to_datetime(df2['time'],errors = 'coerce').dt.normalize() #Also did this for the weather data

#Choosing the columns for the dataframes
selectedCols = ['Activity ID', 'Activity Date', 'Activity Type','Elapsed Time', 'Distance', 'Max Heart Rate',
       'Relative Effort', 'Activity Gear',
       'Filename','Max Speed', 'Average Speed',
       'Elevation Gain', 'Elevation Loss', 'Elevation Low', 'Elevation High',
       'Max Cadence', 'Average Cadence', 'Average Heart Rate']
selectedCols2 = ['time', 'temperature_2m_mean (°F)', 'apparent_temperature_mean (°F)',
       'precipitation_sum (inch)', 'snowfall_sum (inch)', 'rain_sum (inch)',
       'wind_speed_10m_max (mp/h)', 'wind_gusts_10m_max (mp/h)',
       'daylight_duration (s)', 'relative_humidity_2m_mean (%)',
       'weather_code (wmo code)']

df = df[selectedCols]
df2 = df2[selectedCols2]

#Filtered the data to only include Runs (I had some biking and walking activities) as well as runs that occurred after 12/3/2022. I got a Garmin after this so my runs would be a bit more accurate
df = df[df['Activity Type'] == 'Run']
df = df[df['Activity Date'] >= '2022-12-3'] 
df2 = df2[df2['time'] >= '12/3/2022'] #This is to filter the weather data

#Created new features
#Year
df['Year'] = df['Activity Date'].dt.year
#Distance(in Miles). Originally the data uses kilometres
df['Distance in Miles']=df['Distance']/1.609
#Created a Pace column using miles and elapsed time
df['Elapsed Time in Minutes']=df['Elapsed Time']/60 #Originally the data has this time in seconds
df['Pace']=df['Elapsed Time in Minutes']/df['Distance in Miles']

#Making another dataframe for my Distance Per Month Graph
df['Year Month'] = df['Activity Date'].dt.to_period('M')
dfMonthly = df.groupby('Year Month')['Distance in Miles'].sum().reset_index()
dfMonthly['Year Month'] = dfMonthly['Year Month'].dt.to_timestamp()

#Combining the running and weather data
mergedDf=pd.merge(df,df2, left_on='Activity Date', right_on='time', how='left')

'''
#GRAPHS
#Running Distance Per Month
plt.figure(figsize=(14, 6))
plt.bar(dfMonthly['Year Month'], dfMonthly['Distance in Miles'], width=20, color='skyblue')
plt.title("Total Distances Ran Per Month", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Distance (in Miles)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.show()


#Scatter plot for time vs relative effort, where: x=Date, y=Relative Effort, size=Distance
plt.figure(figsize=(14,7))
scatter = sns.scatterplot(data=df, x='Activity Date', y='Relative Effort', size='Distance in Miles',
    alpha=0.7,
    sizes=(20, 200)  #Adjust min and max size of points
)
plt.title('Relative Effort Progression (point size = Distance in Miles)')
plt.xlabel('Activity Date')
plt.ylabel('Relative Effort')
plt.legend(title='Distance in Miles', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#Scatter plot for pace vs average heart rate, where: x=Pace, y=avg hr, size=Distance, color=Year
plt.figure(figsize=(14,7))
scatter = sns.scatterplot(data=df, x='Pace', y='Average Heart Rate', size='Distance',
    hue='Year',
    palette='viridis',
    alpha=0.7,
    sizes=(20, 200)  # adjust min and max size of points
)

plt.title('Pace vs Heart Rate (point size = Distance)')
plt.xlabel('Pace (minutes per mile)')
plt.ylabel('Average Heart Rate')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#Box plot for pace each year
plt.figure(figsize=(14,7))
box = sns.boxplot(data=df, x='Year', y='Pace', fill=True)
plt.title('Paces Each Year')
plt.xlabel('Year')
plt.ylabel('Pace (minutes per mile)')
plt.tight_layout()
plt.show()

#Scatter plot for apparent temperature vs pace where: x=pace, y=avg hr, size=Distance, color=Year
plt.figure(figsize=(14,7))
scatter = sns.scatterplot(data=mergedDf, x='apparent_temperature_mean (°F)', y='Pace', size='Distance',
    hue='Year',
    palette='viridis',
    alpha=0.7,
    sizes=(20, 200)  # adjust min and max size of points
)

plt.title('Apparent Temperature and Pace (point size = Distance)')
plt.xlabel('Apparent Temperature')
plt.ylabel('Pace (minutes per mile)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#Scatter plot for precipation and pace where: x=precipitation, y=pace, size=Distance
plt.figure(figsize=(14,7))
scatter = sns.scatterplot(data=mergedDf, x='precipitation_sum (inch)', y='Pace', size='Distance',
    hue='Year',
    palette='viridis',
    alpha=0.7,
    sizes=(20, 200)  # adjust min and max size of points
)

plt.title('Precipitation and Pace (point size = Distance)')
plt.xlabel('Precipitation (in inches)')
plt.ylabel('Pace (minutes per mile)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
'''

#Linear Regression Analysis
#Do any of these features play important factors in determining the pace?
features=['Distance in Miles','Elevation Gain',
          'Elevation Loss','apparent_temperature_mean (°F)',
          'precipitation_sum (inch)','relative_humidity_2m_mean (%)']
X=mergedDf[features]
y=mergedDf['Pace']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Training model
model = LinearRegression()
model.fit(X_train, y_train)
#Predict
y_pred = model.predict(X_test)
#Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Print coefficients with feature names
print("\n Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f} (min/mile per unit)")

print(f"\nIntercept (Baseline pace): {model.intercept_:.4f} min/mile")
#mergedDf.to_csv('merge.csv', index=False)

#Significance
# X is your feature matrix, y is your target variable (pace)
X = mergedDf[features]
y = mergedDf['Pace']

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())
