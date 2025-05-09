Elbow Method (left plot):
The “elbow” is the point where the distortion (within-cluster sum of squares) starts to decrease more slowly. In your plot, the curve bends most noticeably around k = 7 or 8. After that, the reduction in distortion is less significant.

Silhouette Analysis (right plot):
The silhouette score measures how well-separated the clusters are (higher is better). In your plot, the highest silhouette score is at k = 2, but the value is quite low overall, and there is a local maximum at k = 8.

Interpretation:
The elbow method suggests 7 or 8 clusters.
The silhouette score is highest at k = 2, but the value is very low, indicating weak clustering structure. The next best is at k = 8.

Recommendation:
Given both plots, k = 8 is a reasonable choice. It is at the elbow and also a local maximum in the silhouette score. However, the low silhouette values overall suggest that the data may not cluster very well, or that the features used may not be highly separable.

So this is why we chose to use 8 clusters

