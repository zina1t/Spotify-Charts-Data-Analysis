import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import calendar

# Read data from CSV file
df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

# Drop rows with missing values in the 'key' column
df = df.dropna(subset=['key'])
df.dropna(inplace=True)
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Create a dictionary to map track names to unique identifiers
track_name_to_id = {name: index for index, name in enumerate(df['track_name'].unique())}
df['track_id'] = df['track_name'].map(track_name_to_id)
df.drop(columns=['track_name'], inplace=True)

# Plot a histogram of the number of tracks over the years
plt.figure(figsize = (10, 7))
years = df['released_year'].value_counts().sort_index()
years.plot(x = years.index, y = years, width=0.6, kind='bar')
plt.show()

# Plot a pie chart of the number of tracks over the months
months = df['released_month'].value_counts()
m_names = [calendar.month_name[i] for i in months.index]
plt.figure(figsize = (10, 7))
plt.pie(months, labels=m_names, autopct='%.1f%%')
plt.show()

# Plot a histogram of the number of tracks based on the number of artists
plt.figure(figsize = (10, 7))
artists = df['artist_count'].value_counts().sort_index()
artists.plot(x = artists.index, y = artists, width=0.6, kind='bar', rot=0)
plt.show()

# Plot histograms for various musical characteristics
columns_analysis = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
plt.figure(figsize = (10, 7))
for i, column in enumerate(columns_analysis, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=column, bins=15, color='blue')
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Score", fontsize=12)
plt.tight_layout()
plt.show()

# Calculate and visualize the correlation matrix
correlation_matrix = df[columns_analysis].corr()
correlation_matrix
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, vmin=-0.58, vmax=1, annot=True)


# Plot the impact of various characteristics on the number of streams
fig, axes = plt.subplots(3, 3, figsize=(10, 7))
axes = axes.flatten()
scaler = MinMaxScaler()
df['streams'] = scaler.fit_transform(df[['streams']])

for i, column in enumerate(columns_analysis):
    plt.sca(axes[i])
    plt.bar(df[column], df['streams'], color='blue')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Streams', fontsize=12)
    plt.title(f'Streams vs. {column}', fontsize=14)
    plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Plot a boxplot for the distribution of streams by keys
sorted_keys = sorted(df['key'].unique())
average_streams = df.groupby('key')['streams'].mean().sort_values().index.tolist()
plt.figure(figsize=(10,7))
sns.boxplot(x='streams', y='key', data= df, order=average_streams)
plt.xlabel('Streams')
plt.ylabel('Keys')
plt.title('Box Plot of Streams by Keys')
plt.show()

# Plot a histogram of the count of tracks by keys
plt.figure(figsize=(10, 7))
key_count = df['key'].value_counts(ascending=True)
colors  = sns.color_palette("Set2", len(key_count))
key_count.plot(x = 'key', y = key_count, kind='bar', color=colors)
plt.xlabel('Keys')
plt.ylabel('Count')
plt.title('Count of each key')
plt.xticks(rotation=0)
plt.show()

# Create a DataFrame 'key_df' to analyze the relationship between 'key' and 'streams'
key_df = df[['key', 'streams']].copy()
# Group by 'key' and calculate mean, min, and max values of 'streams'
key_df = key_df.groupby('key')['streams'].agg(['mean', 'min', 'max'])
key_df = key_df.rename(columns={'mean' : 'avg_streams', 'min' : 'min_streams', 'max' : 'max_streams'})

# Plot a bar chart for average streams by key
plt.figure(figsize=(10, 7))
key_df.plot(y ='avg_streams', color=colors, kind='bar', legend=False)
plt.xlabel('Keys')
plt.ylabel('Average streams')
plt.title('Average streams by key')
plt.xticks(rotation=0)
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(10, 7))
axes = axes.flatten()

# Scale the 'streams' column using MinMaxScaler
scaler = MinMaxScaler()
df['streams'] = scaler.fit_transform(df[['streams']])

# Plot bar charts for various musical characteristics vs. 'key'
for i, column in enumerate(columns_analysis):
    plt.sca(axes[i])
    plt.bar(df['key'], df[column])
    plt.ylabel('Key')
    plt.title(f'Key vs. {column}')
    plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Define a list of musical characteristics for scatter plots
columns_analysis = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

# Set the colormap
colormap = plt.colormaps['viridis']

# Plot scatter plots for various musical characteristics vs. 'BPM'
for i, column in enumerate(columns_analysis):
  plt.sca(axes[i])
  scatter = plt.scatter(df['bpm'], df[column], cmap=colormap, c=df[column], alpha=0.5)
  plt.xlabel('BPM')
  plt.ylabel(column)
  plt.title(f'BPM vs. {column}')
  plt.colorbar(scatter, label=column)

 # Remove empty subplots 
[fig.delaxes(ax) for ax in axes if not ax.has_data()] 
plt.tight_layout()
plt.show()

# Repeat the process for other pairs of musical characteristics
# (danceability_% vs. other characteristics, energy_% vs. other characteristics, etc.)

columns_analysis = ['valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
colormap = plt.colormaps['viridis']
for i, column in enumerate(columns_analysis):
  plt.sca(axes[i])
  scatter = plt.scatter(df['danceability_%'], df[column], c=df[column], cmap=colormap, alpha=0.5)
  plt.xlabel('danceability_%')
  plt.ylabel(column)
  plt.title(f'danceability_% vs {column}')
  plt.colorbar(scatter, label=column)
[fig.delaxes(ax) for ax in axes if not ax.has_data()] 
plt.tight_layout()
plt.show()

columns_analysis = ['energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
colormap = plt.colormaps['viridis']
for i, column in enumerate(columns_analysis):
  plt.sca(axes[i])
  scatter = plt.scatter(df['valence_%'], df[column], cmap=colormap, c=df[column], alpha=0.5)
  plt.xlabel('valence_%')
  plt.ylabel(column)
  plt.title(f'valence_% vs {column}')
  plt.colorbar(scatter, label=column)
[fig.delaxes(ax) for ax in axes if not ax.has_data()]
plt.tight_layout()
plt.show()

columns_analysis = ['acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
colormap = plt.colormaps['viridis']
for i, column in enumerate(columns_analysis):
  plt.sca(axes[i])
  scatter = plt.scatter(df['energy_%'], df[column], cmap=colormap, c=df[column], alpha=0.5)
  plt.xlabel('energy_%')
  plt.ylabel(column)
  plt.title(f'energy_% vs {column}')
  plt.colorbar(scatter, label=column)
[fig.delaxes(ax) for ax in axes if not ax.has_data()]
plt.tight_layout()
plt.show()

columns_analysis = ['instrumentalness_%', 'liveness_%', 'speechiness_%']
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
colormap = plt.colormaps['viridis']
for i, column in enumerate(columns_analysis):
  plt.sca(axes[i])
  scatter = plt.scatter(df['acousticness_%'], df[column], cmap=colormap, c=df[column], alpha=0.5)
  plt.xlabel('acousticness_%')
  plt.ylabel(column)
  plt.title(f'acousticness_% vs {column}')
  plt.colorbar(scatter, label=column)
[fig.delaxes(ax) for ax in axes if not ax.has_data()]
plt.tight_layout()
plt.show()

columns_analysis = ['liveness_%', 'speechiness_%']
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
colormap = plt.colormaps['viridis']
for i, column in enumerate(columns_analysis):
  plt.sca(axes[i])
  scatter = plt.scatter(df['instrumentalness_%'], df[column], cmap=colormap, c=df[column], alpha=0.5)
  plt.xlabel('instrumentalness_%')
  plt.ylabel(column)
  plt.title(f'instrumentalness_% vs {column}')
  plt.colorbar(scatter, label=column)
[fig.delaxes(ax) for ax in axes if not ax.has_data()]
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 7))

colormap = plt.colormaps['viridis']
scatter = plt.scatter(df['liveness_%'], df['speechiness_%'], cmap=colormap, c=df[column], alpha=0.5)
plt.xlabel('instrumentalness_%')
plt.ylabel(column)
plt.title(f'instrumentalness_% vs {column}')
plt.colorbar(scatter, label=column)
plt.show()

df_features = df
df_features = pd.get_dummies(df, columns=['key', 'mode'], prefix=['key', 'mode'])
df_features.head()

# Convert the binary columns to integer
binary_columns = ['key_A', 'key_A#', 'key_B', 'key_C#', 'key_D', 'key_D#', 'key_E', 'key_F', 'key_F#', 'key_G', 'key_G#', 'mode_Major', 'mode_Minor']
df1 = df_features
for column in binary_columns:
    df1[column] = df_features[column].astype(int) 

# Drop unnecessary columns
df1.drop(columns=['artist_count', 'track_id','released_year','released_month', 'released_day', 'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_charts'], inplace=True)

# Analyze the top 10, top 50, top 100, top 250, top 500, top 817 popular tracks
top_10_songs = df1.nlargest(10, 'streams')
top_10_songs.describe()

top_50_songs = df1.nlargest(50, 'streams')
top_50_songs.describe()

top_100_songs = df1.nlargest(100, 'streams')
top_100_songs.describe()

top_250_songs = df1.nlargest(250, 'streams')
top_250_songs.describe()

top_500_songs = df1.nlargest(500, 'streams')
top_500_songs.describe()

top_817_songs = df1.nlargest(817, 'streams')
top_817_songs.describe()

# Map each key to a corresponding color
key_colors = {
    'key_A' : 'red',
    'key_A#' : 'orange',
    'key_B' : 'yellow',
    'key_C#' : 'green',
    'key_D' : 'blue',
    'key_D#' : 'purple',
    'key_E' : 'lime',
    'key_F' : 'pink',
    'key_F#' : 'brown',
    'key_G' : 'magenta',
    'key_G#' : 'cyan'
}

key_columns = ['key_A', 'key_A#', 'key_B', 'key_C#', 'key_D', 'key_D#', 'key_E', 'key_F', 'key_F#', 'key_G', 'key_G#']

chart_titles = ['Top 10 Songs Distribution', 'Top 50 Songs Distribution', 'Top 100 Songs Distribution']
dataframes = [top_10_songs, top_50_songs, top_100_songs]

fig, axes = plt.subplots(1, len(dataframes), figsize=(15, 5))

# create pie charts for the top 10, top 50, top 100 popular tracks based on keys, without labels and with the legend
for i, df in enumerate(dataframes):
    plt.sca(axes[i])
    key_counts = df[key_columns].eq(1).sum()
    key_counts_filtered = key_counts[key_counts > 0]
    plt.pie(key_counts_filtered, labels=None, startangle=140, autopct='%.1f%%', colors=[key_colors[key] for key in key_counts_filtered.index])
    plt.title(chart_titles[i])
legend_labels = [key for key in key_counts_filtered.index]
plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()

#key_C# is the most popular key in the top 10, top 50, top 100 popular tracks
key_of_interest = 'key_C#'

top_dataframes = [top_10_songs, top_50_songs, top_100_songs, top_250_songs, top_500_songs, top_817_songs]
labels=['Top 10', 'Top 50', 'Top 100', 'Top 250', 'Top 500', 'Top 817']
key_percentage = []

# Calculate the percentage of key_C# in the top 10, top 50, top 100, top 250, top 500, top 817 popular tracks
for df in top_dataframes:
    key_counts = df[key_columns].eq(1).sum()
    key_percentage.append(key_counts[key_of_interest] / key_counts.sum() * 100)

# Plot a bar chart for the percentage of key_C# in the top 10, top 50, top 100, top 250, top 500, top 817 popular tracks
plt.figure(figsize=(10, 7))
plt.bar(labels, key_percentage, color='skyblue')
plt.xlabel('Number of Songs')
plt.ylabel('% of key_C#')
plt.title('Percentage of key_C# in Songs')

# Add labels to the bar chart
for i, percentage in enumerate(key_percentage):
  plt.text(i, percentage, f'{percentage:.2f}%', ha='center', va='bottom')
plt.show()

# Analyze the last 10, last 50, last 100 popular tracks
last_10_songs = df1.nsmallest(10, 'streams')
last_50_songs = df1.nsmallest(50, 'streams')
last_100_songs = df1.nsmallest(100, 'streams')

last_dataframes = [last_10_songs, last_50_songs, last_100_songs]
last_labels=['Last 10', 'Last 50', 'Last 100']
key_percentage = []

for df in last_dataframes:
    key_counts = df[key_columns].eq(1).sum()
    key_percentage.append(key_counts[key_of_interest] / key_counts.sum() * 100)

# Plot a bar chart for the percentage of key_C# in the last 10, last 100, last 250 popular tracks
plt.figure(figsize=(10, 7))
plt.bar(last_labels, key_percentage, color='skyblue')
plt.xlabel('Number of Songs')
plt.ylabel('% of key_C#')
plt.title('Percentage of key_C# in Songs')

# Add labels to the bar chart
for i, percentage in enumerate(key_percentage):
  plt.text(i, percentage, f'{percentage:.2f}%', ha='center', va='bottom')
plt.show()

# Count major, minor percentage in top_dataframes
top_mode_major_percentage = [(df['mode_Major'].eq(1).sum() / len(df) * 100) for df in top_dataframes]
top_mode_minor_percentage = [(df['mode_Minor'].eq(1).sum() / len(df) * 100) for df in top_dataframes]

# Plot a stacked bar chart for the number of songs in major and minor keys in the top 10, top 50, top 100, top 250, top 500, top 817 popular tracks
plt.figure(figsize=(10, 7))
plt.bar(labels, top_mode_major_percentage, color='skyblue')
plt.bar(labels, top_mode_minor_percentage, bottom=top_mode_major_percentage, color='orange')
plt.xlabel('Number of Songs')
plt.ylabel('% of mode')
plt.title('% of Songs in Major and Minor Keys')
plt.legend(['Major', 'Minor'])
plt.yticks(np.arange(0, 110, 10))
plt.show()

# Plot chart for the minimum, average, maximum, and standard deviation of BPM in the popular tracks
bpm_min = [df['bpm'].min() for df in dataframes]
bpm_avg = [df['bpm'].mean() for df in dataframes]
bpm_max = [df['bpm'].max() for df in dataframes]
bpm_std = [df['bpm'].std() for df in dataframes]

plt.figure(figsize=(10, 7))
plt.plot(labels, bpm_max, color='green', marker='o')
plt.plot(labels, bpm_avg, color='blue', marker='o')
plt.plot(labels, bpm_min, color='orange', marker='o',)
plt.plot(labels, bpm_std, color='red', marker='o')
plt.xlabel('Number of Songs')
plt.ylabel('BPM')
plt.title('Distribution of BPM in Popular Songs')
plt.legend(['Maximum', 'Average', 'Minimum', 'Standard Deviation'], loc='upper right', bbox_to_anchor=(1.0, 0.9))

#add text labels to the lines
for i, (max_val, avg_val, min_val, std_val) in enumerate(zip(bpm_max, bpm_avg, bpm_min, bpm_std)):
  plt.text(i, max_val, f'{max_val:.0f}', ha='center', va='bottom')
  plt.text(i, avg_val, f'{avg_val:.0f}', ha='center', va='bottom')
  plt.text(i, min_val, f'{min_val:.0f}', ha='center', va='bottom')
  plt.text(i, std_val, f'{std_val:.0f}', ha='center', va='bottom')
plt.show()
