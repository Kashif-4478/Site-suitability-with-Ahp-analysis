# Site-Suitability-with-AHP-Analysis

# Site Suitability for schools

This project is designed to aid educational planners, government officials, and community stakeholders in identifying optimal locations for the construction of new schools. By leveraging Geographic Information System (GIS) techniques and the Analytic Hierarchy Process (AHP), the project analyzes various criteria such as infrastructure, population density, and land use to determine the most suitable sites. The aim is to facilitate informed decision-making in school site selection, ensuring that resources are allocated efficiently and that new schools contribute to the overall development and accessibility of education within a region.


## Libraries


Libraries such as Rasterio and Matplotlib play a crucial role in this project by providing essential tools for efficiently handling geospatial raster data and creating visualizations. These libraries empower the project to conduct informed analyses and visually communicate complex geographical information, ultimately enhancing the decision-making process in the selection of optimal school construction sites. The use of specialized libraries contributes to the project's effectiveness in leveraging spatial data and making data-driven decisions.






```bash
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.plot import show
```

## Displaying Datasets


The datatsets were obtained from different sources and preprocessed before usage to get reclassified raster datasets which are crucial for the site suitability for assigning criteria weights
```bash
# Specify the paths to the reclassified raster files
roads_file = '/content/drive/MyDrive/extracted_folder/Outputs/rec_r'
schools_file = '/content/drive/MyDrive/extracted_folder/Outputs/rec_s'
water_file = '/content/drive/MyDrive/extracted_folder/Outputs/rec_w'
land_use_file = '/content/drive/MyDrive/extracted_folder/Outputs/rec_lulc'
population_density_file = '/content/drive/MyDrive/extracted_folder/Outputs/rec_pop'
slope_file = '/content/drive/MyDrive/extracted_folder/Outputs/rec_slope'

# Open each reclassified raster file
roads_data = rasterio.open(roads_file)
schools_data = rasterio.open(schools_file)
water_data = rasterio.open(water_file)
land_use_data = rasterio.open(land_use_file)
population_density_data = rasterio.open(population_density_file)
slope_data = rasterio.open(slope_file)

# Plot all reclassified raster layers
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

show(roads_data, ax=axs[0, 0], cmap='Blues', title='Reclassified Roads')
show(schools_data, ax=axs[0, 1], cmap='Greens', title='Reclassified Schools')
show(water_data, ax=axs[0, 2], cmap='Blues', title='Reclassified Water')
show(land_use_data, ax=axs[1, 0], cmap='tab20', title='Reclassified Land Use')
show(population_density_data, ax=axs[1, 1], cmap='YlOrRd', title='Reclassified Population Density')
show(slope_data, ax=axs[1, 2], cmap='viridis', title='Reclassified Slope')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

# Close all raster files
roads_data.close()
schools_data.close()
water_data.close()
land_use_data.close()
population_density_data.close()
slope_data.close()

```
![image](https://github.com/Kashif-4477/Site-Suitability-Analysis-Using-Ahp/assets/98015675/c48d62a2-568f-4524-b0bf-0144bb6a9887)
## Reading rasters using rasterio and building Pairwise Comparison Matrix
The rasterio library is crucial for handleling raster datasets such as reading and displaing the datasets after that for the implicaton of comparison which is pivotal in decision-making, providing a structured framework for quantifying subjective preferences, facilitating pairwise comparisons, and ensuring consistency. In methodologies like AHP, it plays a crucial role in deriving weights, contributing to transparent and robust decision outcomes

```bash
with rasterio.open(slope_file) as src1, rasterio.open(schools_file) as src2, rasterio.open(water_file) as src3, rasterio.open(population_density_file) as src4, rasterio.open(land_use_file) as src5, rasterio.open(roads_file) as src6:
    raster1 = src1.read(1)
    raster2 = src2.read(1)
    raster3 = src3.read(1)
    raster4 = src4.read(1)
    raster5 = src5.read(1)
    raster6 = src6.read(1)
import numpy as np

# Pairwise comparison matrix for school site selection criteria
pairwise_matrix = np.array([
    [1, 1/3, 1/2, 1/4, 1/7, 1/5],    # Slope
    [3, 1, 2, 1/2, 3, 1/2],       # Schools
    [2, 1/2, 1, 1/3, 1/6, 1/4],      # Water
    [4, 2, 3, 1, 7, 2/3],          # Population Density
    [7, 1/3, 6, 1/7, 1, 1/6],        # Land Use
    [5, 2, 4, 3/2, 6, 1]             # Roads
])

# Step 1: Normalize each column
normalized_matrix = pairwise_matrix / pairwise_matrix.sum(axis=0)

# Step 2: Calculate the average for each row (weights)
weights = normalized_matrix.mean(axis=1)

# Other axis points
other_axis_points = np.array([1, 2, 3, 4, 5, 6])

# Calculate values for other axis using weights
values_for_other_axis = weights * other_axis_points

# Print the calculated values
print("Values for Other Axis:")
print(values_for_other_axis)

```

## Ahp Analysis


AHP analysis is pivotal in our project for school site selection, providing a structured framework to systematically prioritize criteria. It transforms subjective judgments into quantifiable data, ensures consistency, and calculates weighted priorities for informed decision-making, facilitating a comprehensive evaluation of diverse factors.







```bash

# AHP weights for each class (normalized)
ahp_weights_raster1 = np.array([0.5, 0.2, 0.1, 0.2, 0.1, 0.3])
ahp_weights_raster2 = np.array([0.8, 0.3, 0.4, 0.1, 0.2, 0.2])
ahp_weights_raster3 = np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.1])
ahp_weights_raster4 = np.array([0.2, 0.2, 0.1, 0.2, 0.2, 0.4])
ahp_weights_raster5 = np.array([0.2, 0.2, 0.4, 0.1, 0.2, 0.1])
ahp_weights_raster6 = np.array([0.3, 0.1, 0.1, 0.2, 0.6, 0.9])

# Weighted sum for each raster
weighted_sum_raster1 = np.sum(ahp_weights_raster1[:, np.newaxis, np.newaxis] * raster1, axis=0)
weighted_sum_raster2 = np.sum(ahp_weights_raster2[:, np.newaxis, np.newaxis] * raster2, axis=0)
weighted_sum_raster3 = np.sum(ahp_weights_raster3[:, np.newaxis, np.newaxis] * raster3, axis=0)
weighted_sum_raster4 = np.sum(ahp_weights_raster4[:, np.newaxis, np.newaxis] * raster4, axis=0)
weighted_sum_raster5 = np.sum(ahp_weights_raster5[:, np.newaxis, np.newaxis] * raster5, axis=0)
weighted_sum_raster6 = np.sum(ahp_weights_raster6[:, np.newaxis, np.newaxis] * raster6, axis=0)

# The results are the weighted sums for each pixel in the raster based on AHP weights
import rasterio
from rasterio.enums import Resampling
import numpy as np

# Open the rasters
with rasterio.open(slope_file) as src1, rasterio.open(schools_file) as src2, \
        rasterio.open(water_file) as src3, rasterio.open(population_density_file) as src4, \
        rasterio.open(land_use_file) as src5, rasterio.open(roads_file) as src6:

    raster1 = src1.read(1)

    # Resample rasters to have the same shape
    shape = src1.shape  # Use the shape of one raster as the target shape
    raster2 = src2.read(1, out_shape=shape, resampling=Resampling.nearest)
    raster3 = src3.read(1, out_shape=shape, resampling=Resampling.nearest)
    raster4 = src4.read(1, out_shape=shape, resampling=Resampling.nearest)
    raster5 = src5.read(1, out_shape=shape, resampling=Resampling.nearest)
    raster6 = src6.read(1, out_shape=shape, resampling=Resampling.nearest)

    # Assuming each raster has 6 classes
    num_classes = 6

    # Reshape AHP weights
    ahp_weights_raster1 = ahp_weights_raster1.reshape((1, 1, num_classes))
    ahp_weights_raster2 = ahp_weights_raster2.reshape((1, 1, num_classes))
    ahp_weights_raster3 = ahp_weights_raster3.reshape((1, 1, num_classes))
    ahp_weights_raster4 = ahp_weights_raster4.reshape((1, 1, num_classes))
    ahp_weights_raster5 = ahp_weights_raster5.reshape((1, 1, num_classes))
    ahp_weights_raster6 = ahp_weights_raster6.reshape((1, 1, num_classes))

    # Expand dimensions of raster to match the shape of AHP weights
    raster1_expanded = np.expand_dims(raster1, axis=-1)
    raster2_expanded = np.expand_dims(raster2, axis=-1)
    raster3_expanded = np.expand_dims(raster3, axis=-1)
    raster4_expanded = np.expand_dims(raster4, axis=-1)
    raster5_expanded = np.expand_dims(raster5, axis=-1)
    raster6_expanded = np.expand_dims(raster6, axis=-1)

    # Calculate AHP Weighted Sum
    weighted_sum_raster1 = np.sum(raster1_expanded * ahp_weights_raster1, axis=-1)
    weighted_sum_raster2 = np.sum(raster2_expanded * ahp_weights_raster2, axis=-1)
    weighted_sum_raster3 = np.sum(raster3_expanded * ahp_weights_raster3, axis=-1)
    weighted_sum_raster4 = np.sum(raster4_expanded * ahp_weights_raster4, axis=-1)
    weighted_sum_raster5 = np.sum(raster5_expanded * ahp_weights_raster5, axis=-1)
    weighted_sum_raster6 = np.sum(raster6_expanded * ahp_weights_raster6, axis=-1)

    # Sum the weighted rasters
    ahp_suitability_map = weighted_sum_raster1 + weighted_sum_raster2 + weighted_sum_raster3 + \
                           weighted_sum_raster4 + weighted_sum_raster5 + weighted_sum_raster6

```

## Results

The map classifies each potential school site as either highly suitable, moderately suitable, or least suitable. The most suitable sites are those that are closest to population centers, have the best access to transportation, and are the safest. The least suitable sites are those that are the furthest from population centers, have the worst access to transportation, and are the least safe.

```bash
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Assuming ahp_suitability_map is your data

# Create a colormap
cmap = plt.get_cmap('hot')
cmap.set_under('white')

# Plot the image with positive values only
plt.figure(figsize=(10, 10))
img = plt.imshow(ahp_suitability_map, cmap=cmap, extent=src1.bounds, vmin=15, vmax=80)
plt.colorbar(img, label='AHP Suitability')

# Create a legend using Line2D objects
legend_elements = [
    Line2D([0], [0], marker='s', color=cmap(0), markerfacecolor=cmap(0), markersize=10, label='Most Suitable'),
    Line2D([0], [0], marker='s', color=cmap(0.3), markerfacecolor=cmap(0.3), markersize=10, label='Moderately Suitable'),
    Line2D([0], [0], marker='s', color=cmap(1.0), markerfacecolor=cmap(1.0), markersize=10, label='Least Suitable')
]

plt.legend(handles=legend_elements, title='Suitability Classes', loc='upper left')
plt.title('Site Suitability Map (AHP)')

# Show the plot
plt.show()

```

![image](https://github.com/Kashif-4477/Site-Suitability-Analysis-Using-Ahp/assets/98015675/7dd00af7-873e-4563-8d4c-cef9e3221656)
