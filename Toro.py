import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def shade_area(start, end):
    if start < 0 or end >= len(X) or start >= end:
        print("Invalid start or end index.")
        return
    
    polygon_points = [(0, 0)]
    polygon_points += [(X[i], Y[i]) for i in range(start, end + 1)]
    polygon_points.append((0, 0))
    
    polygon = plt.Polygon(polygon_points, color='orange', alpha=0.4, label="Area")
    plt.gca().add_patch(polygon)

def calculate_polygon_area(start, end):
    if start < 0 or end >= len(X) or start >= end:
        print("Invalid start or end index.")
        return None
    
    polygon_points = [(0, 0)]
    polygon_points += [(X[i], Y[i]) for i in range(start, end + 1)]
    polygon_points.append((0, 0))
    n = len(polygon_points)
    area = 0.5 * abs(sum(polygon_points[i][0] * polygon_points[i+1][1] - polygon_points[i+1][0] * polygon_points[i][1]
                        for i in range(n - 1)))
    
    return area

angles_degrees = np.array([3.0, 20.0, 43.0, 78.0, 126.5, 175.0, 210.0, 233.0, 250.0, 
        263.0, 274.0, 283.5, 292.0, 300.5, 308.5, 316.5, 325.0, 
        334.5, 344.5, 356.5])
distances_au = np.array([1.46, 1.26, 1.05, 0.86, 0.77, 0.86, 1.05, 1.26, 1.46, 
        1.62, 1.75, 1.85, 1.92, 1.96, 1.96, 1.94, 1.89, 1.80, 
        1.69, 1.54])

angles_radians = np.radians(angles_degrees)
X = distances_au * np.cos(angles_radians)
Y = distances_au * np.sin(angles_radians)

data = np.column_stack((X, Y))
pca = PCA(n_components=2)
pca.fit(data)

transformed_data = pca.transform(data)
center_transformed = (transformed_data.max(axis=0) + transformed_data.min(axis=0)) / 2
center = pca.inverse_transform(center_transformed)
major_axis = pca.components_[0] * 1.36
minor_axis = pca.components_[1] * 1.2

major_axis_length = np.ptp(transformed_data[:, 0]) / 2
minor_axis_length = np.ptp(transformed_data[:, 1]) / 2

a = np.linalg.norm(major_axis)
b = np.linalg.norm(minor_axis)
c = np.sqrt(a**2 - b**2)
foci1 = center + major_axis * (c / a)
foci2 = center - major_axis * (c / a)

aphelion_point = center + pca.components_[0] * major_axis_length
perihelion_point = center - pca.components_[0] * major_axis_length

foci_distance_from_center = np.sqrt((major_axis_length/2)**2 - (minor_axis_length/2)**2)
focus1 = center + foci_distance_from_center * (major_axis / np.linalg.norm(major_axis))
focus2 = center - foci_distance_from_center * (major_axis / np.linalg.norm(major_axis))
eccentricity = foci_distance_from_center / (major_axis_length/2)
print(f"Eccentricity: {eccentricity}")

plt.figure(figsize=(10, 8))
plt.plot(X, Y, 'o-', label='Orbital Path', color='b')
plt.title("Asteroid Toro's Orbital Path")
plt.xlabel("X (AU)")
plt.ylabel("Y (AU)")
plt.grid(True)
plt.axis('equal')

plt.plot([center[0] + major_axis[0], center[0] - major_axis[0]], 
         [center[1] + major_axis[1], center[1] - major_axis[1]], color='r', label="Major Axis")
plt.plot([center[0] + minor_axis[0], center[0] - minor_axis[0]], 
         [center[1] + minor_axis[1], center[1] - minor_axis[1]], color='g', label="Minor Axis")

plt.scatter(foci1[0], foci1[1], color='purple', s=100, marker='o', label="Focus 1", zorder=2)
plt.scatter(foci2[0], foci2[1], color='orange', s=100, marker='o', label="Focus 2", zorder=2)

plt.scatter(aphelion_point[0], aphelion_point[1], color='cyan', s=150, edgecolor='black', marker='o', label="Aphelion", zorder=2)
plt.scatter(perihelion_point[0], perihelion_point[1], color='magenta', s=150, edgecolor='black', marker='o', label="Perihelion", zorder=2)

shade_area(3, 5)
shade_area(12, 15)

far_area = calculate_polygon_area(3, 5)
close_area = calculate_polygon_area(13, 15)

print(f"Far area: {far_area}")
print(f"Close area: {close_area}")

plt.legend()
plt.savefig("Toro.png")
plt.show()

