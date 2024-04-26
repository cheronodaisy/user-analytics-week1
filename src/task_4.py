# %%
"""
This module contains functions for data preprocessing and analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.cluster import KMeans

# %%
load_dotenv()

password = os.getenv('PG_PASSWORD')
# %%


DATABASE_NAME = 'telcom'
TABLE_NAME= 'xdr_data'

connection_params = { "host": "localhost", "user": "postgres", "password": password,
                    "port": "5432", "database": DATABASE_NAME}

engine = create_engine(f"postgresql+psycopg2://{connection_params['user']}:
                       {connection_params['password']}@
                       {connection_params['host']}:
                       {connection_params['port']}/{connection_params['database']}")

SQL_QUERY = 'SELECT * FROM xdr_data'

df = pd.read_sql(SQL_QUERY, con= engine)


# %%
df.to_csv('df.csv', index=False)

# %%
df.info()

# %% [markdown]
# 4.1

# %% [markdown]
# Handle missing values, replace with median for numerical, replace with mode for categorical

# %%
columns_to_fill_median = ['Dur. (ms)', 'IMSI', 'MSISDN/Number', 'IMEI',
                        'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                        'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
                        'TCP DL Retrans. Vol (Bytes)',
                        'TCP UL Retrans. Vol (Bytes)', 'DL TP < 50 Kbps (%)', 
                        '50 Kbps < DL TP < 250 Kbps (%)',
                        '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)', 
                        'UL TP < 10 Kbps (%)',
                        '10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)', 
                        'UL TP > 300 Kbps (%)',
                        'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)', 
                        'Activity Duration UL (ms)',
                        'Dur. (ms).1', 'Nb of sec with 125000B < Vol DL', 
                        'Nb of sec with 1250B < Vol UL < 6250B',
                        'Nb of sec with 31250B < Vol DL < 125000B', 
                        'Nb of sec with 37500B < Vol UL',
                        'Nb of sec with 6250B < Vol DL < 31250B', 
                        'Nb of sec with 6250B < Vol UL < 37500B',
                        'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B', 
                        'Social Media DL (Bytes)',
                        'Social Media UL (Bytes)', 'Netflix DL (Bytes)',
                        'Netflix UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 
                        'Email DL (Bytes)',
                        'Email UL (Bytes)', 'Gaming DL (Bytes)', 
                        'Gaming UL (Bytes)',
                        'Total DL (Bytes)', 'Total UL (Bytes)']

column_to_fill_mode = ['Start', 'End', 'Last Location Name', 'Handset Type', 'Handset Manufacturer']

df[columns_to_fill_median] = df[columns_to_fill_median].fillna(df[columns_to_fill_median].median())

df[column_to_fill_mode] = df[column_to_fill_mode].fillna(df[column_to_fill_mode].mode().iloc[0])

cleaned_df = pd.concat([df[columns_to_fill_median], df[column_to_fill_mode]], axis=1)

# %%
# cleaned_df.to_csv('cleaned_dataset.csv', index=False)

# %%
cleaned_df.isna().sum()

# %% [markdown]
# Aggregate, per customer

# %%
aggregated_df = cleaned_df.groupby('MSISDN/Number').agg({'TCP DL Retrans. Vol (Bytes)': 'first',
                                                 'TCP UL Retrans. Vol (Bytes)': 'first',
                                                 'Avg RTT DL (ms)': 'first',
                                                 'Avg RTT UL (ms)': 'first',
                                                 'Avg Bearer TP DL (kbps)': 'first',
                                                 'Avg Bearer TP UL (kbps)': 'first',
                                                 'Handset Type': 'first'  
                                                 }).reset_index()

print(aggregated_df)


# %% [markdown]
# Compute & list 10 of the top, bottom and most frequent:
# TCP values in the dataset.
# RTT values in the dataset.
# Throughput values in the dataset.
#

# %%
top_tcp_values = cleaned_df['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
bottom_tcp_values = cleaned_df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
most_frequent_tcp_values = cleaned_df['TCP DL Retrans. Vol (Bytes)'].value_counts().nlargest(10)

top_rtt_values = cleaned_df['Avg RTT DL (ms)'].nlargest(10)
bottom_rtt_values = cleaned_df['Avg RTT DL (ms)'].nsmallest(10)
most_frequent_rtt_values = cleaned_df['Avg RTT DL (ms)'].value_counts().nlargest(10)

top_throughput_values = cleaned_df['Avg Bearer TP DL (kbps)'].nlargest(10)
bottom_throughput_values = cleaned_df['Avg Bearer TP DL (kbps)'].nsmallest(10)
most_frequent_throughput_values = cleaned_df['Avg Bearer TP DL (kbps)'].value_counts().nlargest(10)

print(top_tcp_values)
print(bottom_tcp_values)
print(most_frequent_tcp_values)
print(top_rtt_values)
print(bottom_rtt_values)
print(most_frequent_rtt_values)
print(top_throughput_values)
print(bottom_throughput_values)
print(most_frequent_throughput_values)


# %% [markdown]
# The average throughput  per handset type & Average TCP retransmission per handset type
#

# %%
avg_throughput_per_handset = cleaned_df.groupby('Handset Type')[['Avg Bearer TP DL (kbps)', 
                                                                 'Avg Bearer TP UL (kbps)']].mean()
top_10_handsets = avg_throughput_per_handset['Avg Bearer TP DL (kbps)'].nlargest(10)

plt.figure(figsize=(12, 6))
top_10_handsets.plot(kind='bar', alpha=0.75)
plt.title('Average Throughput per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput (kbps)')
plt.xticks(rotation=45, ha='right')
plt.legend(['Downlink', 'Uplink'])
plt.grid(axis='y')
plt.show()


# %% [markdown]
# A k-means clustering (where k = 3) to segment users into groups of experiences

# %%


experience_metrics = cleaned_df[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 
                                 'TCP DL Retrans. Vol (Bytes)']]

kmeans = KMeans(n_clusters=3, random_state=42)
cleaned_df['Cluster'] = kmeans.fit_predict(experience_metrics)

