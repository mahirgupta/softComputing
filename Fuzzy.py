!pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Create fuzzy input variables and membership functions
solar_generation = ctrl.Antecedent(np.arange(0, 1001, 1), 'Solar Generation (W)')
demand = ctrl.Antecedent(np.arange(0, 1001, 1), 'Energy Demand (W)')
grid_stability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'Grid Stability (Index)')
price_fluctuation = ctrl.Antecedent(np.arange(5, 11, 0.1), 'Price Fluctuation (Currency)')
weather_condition = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'Weather Condition (Index)')

solar_generation['low'] = fuzz.trimf(solar_generation.universe, [0, 0, 500])
solar_generation['medium'] = fuzz.trimf(solar_generation.universe, [200, 500, 800])
solar_generation['high'] = fuzz.trimf(solar_generation.universe, [500, 1000, 1000])

demand['low'] = fuzz.trimf(demand.universe, [0, 0, 500])
demand['medium'] = fuzz.trimf(demand.universe, [200, 500, 800])
demand['high'] = fuzz.trimf(demand.universe, [500, 1000, 1000])

grid_stability['poor'] = fuzz.trimf(grid_stability.universe, [0, 0, 0.5])
grid_stability['good'] = fuzz.trimf(grid_stability.universe, [0.4, 0.75, 1])

price_fluctuation['low'] = fuzz.trapmf(price_fluctuation.universe, [5, 5, 6.2, 6.8])
price_fluctuation['high'] = fuzz.trapmf(price_fluctuation.universe, [6.2, 6.8, 11, 11])

weather_condition['poor'] = fuzz.trimf(weather_condition.universe, [0, 0, 0.5])
weather_condition['good'] = fuzz.trimf(weather_condition.universe, [0.4, 0.75, 1])

# Visualize the membership functions
solar_generation.view()
demand.view()
grid_stability.view()
price_fluctuation.view()
weather_condition.view()

plt.show()

centroid_value = 50  # Adjust this value based on your system

# Generate a plot for the defuzzification
x = np.arange(0, 101, 1)
y = np.zeros_like(x)
y[x == centroid_value] = 1  # Set the centroid point to 1

plt.plot(x, y, label='Defuzzification Result', linewidth=2)
plt.xlabel('Optimization Level')
plt.ylabel('Membership')
plt.title('Defuzzification Result (Centroid Method)')
plt.legend()
plt.grid(True)
plt.show()
