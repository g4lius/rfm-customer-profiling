import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def analyze_online_retail_data(filename, n_clusters_range):
    # Loading the Dataset
    data = pd.read_csv(filename, encoding='ISO-8859-1', parse_dates=['InvoiceDate'], sep=',')
    print("Dataset dimension:", data.shape)
    print("Describe: " , data.describe())
    print("Columns of dataset: ", data.columns)
    # Delete rows with null values in the 'CustomerID' field
    data = data.dropna(subset=['CustomerID'])
    print("Dataset dimension:", data.shape)
    print("Describe: " , data.describe())
    # RFM feature extraction
    rfm_data = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'count',
        'Quantity': 'sum'
    })

    rfm_data.columns = ['Recency', 'Frequency', 'Monetary']

    # Applying the StandardScalar method
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)

    # Finding the optimal number of clusters using the elbow method and silhoutte score
    elbow_scores = []
    silhouette_scores = []
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(rfm_scaled)
        elbow_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

    # Plot elbow
    plt.plot(n_clusters_range, elbow_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Elbow Score')
    plt.title('Elbow Method')
    plt.show()

    # Plot silhouette
    plt.plot(n_clusters_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.show()

    # Calculation of the optimal number of clusters
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    print("Optimal number of clusters:", optimal_n_clusters)

    # Running the K-means algorithm with the chosen number of clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(rfm_scaled)

    # Calculating the centres of mass of the various clusters
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # Scatter plot of 2D results
    plt.figure(figsize=(10, 6))
    plt.scatter(rfm_data['Recency'], rfm_data['Frequency'], c=kmeans.labels_, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red', label='Cluster Centers')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    plt.title('Cluster Analysis (2D)')
    plt.legend()
    plt.show()

    # Scatter plot of 3D results
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rfm_data['Recency'], rfm_data['Frequency'], rfm_data['Monetary'], c=kmeans.labels_, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', c='red',
               label='Cluster Centers')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('Cluster Analysis (3D)')
    ax.legend()
    plt.show()

    # Distance of each data point from its centre of mass
    distances = kmeans.transform(rfm_scaled)
    min_distances = np.min(distances, axis=1)

    # Searching for and removing outliers
    threshold = np.percentile(min_distances, 95)
    outliers = rfm_data[min_distances > threshold]
    filtered_data = rfm_data[min_distances <= threshold]

    # Outliers statistics
    n_outliers = len(outliers)
    print("Number of outliers:", n_outliers)
    print("Outliers:")
    print(outliers)

    filtered_data = filtered_data.reset_index(drop=True).drop_duplicates()

    # Scatter plot of 2D results without outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_data['Recency'], filtered_data['Frequency'], c=kmeans.labels_[:len(filtered_data)], cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red', label='Cluster Centers')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    plt.title('Cluster Analysis (2D) - Filtered')
    plt.legend()
    plt.show()

    # Scatter plot of 3D results without outliers
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(filtered_data['Recency'], filtered_data['Frequency'], filtered_data['Monetary'], c=kmeans.labels_[:len(filtered_data)], cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', c='red', label='Cluster Centers')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('Cluster Analysis (3D) - Filtered')
    ax.legend()
    plt.show()


    # Filtered data without outliers
    return filtered_data

filtered_data = analyze_online_retail_data("OnlineRetail.csv", range(2, 10))
