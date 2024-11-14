import matplotlib.pyplot as plt
import numpy as np
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

time = np.array(["0h 00m", "0h 15m", "0h 30m", "0h 45m", "1h 00m", "1h 15m", 
        "1h 30m", "1h 45m", "2h 00m", "2h 15m", "2h 30m", "2h 45m", 
        "3h 00m", "3h 15m", "3h 30m", "3h 45m", "4h 00m", "4h 15m", 
        "4h 30m", "4h 45m", "5h 00m", "5h 15m", "5h 30m", "5h 45m", 
        "6h 00m", "6h 15m", "6h 30m", "6h 45m", "7h 00m", "7h 15m", 
        "7h 30m", "7h 45m", "8h 00m", "8h 15m", "8h 30m", "8h 45m", 
        "9h 00m", "9h 15m", "9h 30m", "9h 45m", "10h 00m", "10h 15m", 
        "10h 30m", "10h 45m", "11h 00m", "11h 15m", "11h 30m", "11h 45m"])
X = np.array([-3.62, -3.46, -3.25, -2.97, -2.60, -2.14, -1.55, -0.85, 0.00, 
              0.78, 1.45, 1.87, 2.09, 2.16, 2.11, 1.99, 1.82, 1.61, 1.37, 
              1.11, 0.85, 0.58, 0.28, 0.00, -0.27, -0.56, -0.84, -1.12, 
              -1.38, -1.64, -1.89, -2.14, -2.37, -2.59, -2.80, -2.99, 
              -3.17, -3.33, -3.49, -3.59, -3.69, -3.77, -3.81, -3.83, 
              -3.81, -3.76, -3.65])
Y = np.array([1.04, 0.63, 0.20, -0.22, -0.65, -1.03, -1.37, -1.58, -1.59, 
              -1.32, -0.79, -0.11, 0.58, 1.22, 1.82, 2.35, 2.81, 3.22, 
              3.59, 3.90, 4.16, 4.40, 4.58, 4.74, 4.86, 4.95, 5.01, 
              5.03, 5.04, 5.00, 4.95, 4.87, 4.77, 4.65, 4.50, 4.33, 
              4.14, 3.93, 3.69, 3.42, 3.15, 2.85, 2.52, 2.20, 1.83, 
              1.46, 1.06])

data = np.column_stack((X, Y))
pca = PCA(n_components=2)
pca.fit(data)

transformed_data = pca.transform(data)

center_transformed = (transformed_data.max(axis=0) + transformed_data.min(axis=0)) / 2
center = pca.inverse_transform(center_transformed)

major_axis = pca.components_[0] * 3.4
minor_axis = pca.components_[1] * 2.85

major_end1 = center + major_axis
major_end2 = center - major_axis
minor_end1 = center + minor_axis
minor_end2 = center - minor_axis

plt.figure(figsize=(10, 8))
plt.plot(X, Y, marker='o', linestyle='-', color='b', label="Satellite Path")
plt.title('Satellite Orbit with Major and Minor Axes')
plt.xlabel('X (lunar radii)')
plt.ylabel('Y (lunar radii)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')

plt.plot([major_end1[0], major_end2[0]], [major_end1[1], major_end2[1]], color='r', label="Major Axis")
plt.plot([minor_end1[0], minor_end2[0]], [minor_end1[1], minor_end2[1]], color='g', label="Minor Axis")

shade_area(7, 11)
shade_area(30, 34)
print(f"Far area: {calculate_polygon_area(7, 11)} lunar radii")
print(f"Far area: {calculate_polygon_area(30, 34)} lunar radii")

a = np.sqrt(pca.explained_variance_[0])
b = np.sqrt(pca.explained_variance_[1])

eccentricity = np.sqrt(b**2/a**2)
print(f"Eccentricity: {eccentricity}")

period = time[len(time) - 1]
print(f"Period: {period}")

plt.grid()
plt.axis('equal')
plt.xlim(-4, 4)
plt.ylim(-2, 6)
plt.legend()
plt.savefig("Moon.png")

hours, minutes = period.split("h")
hours = int(hours.strip())
minutes = int(minutes.strip().replace("m", ""))
T = (hours * 3600) + (minutes * 60)

lunar_radius_meters = 1737400
avg_radius = a*lunar_radius_meters
G = 6.67430e-11
M = (4*np.pi*np.pi*avg_radius*avg_radius*avg_radius)/(G*T*T)
print(f"Mass: {M}")