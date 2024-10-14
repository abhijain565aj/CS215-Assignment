import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data, skipping metadata rows
data = pd.read_csv(
    './data/1/NED21.11.1-D-5.1.1-20111026-v20120814.csv', skiprows=12)

# the first 1500 rows and filter rows where distance (D (Mpc)) < 4
filtered_data = data.loc[:1499, 'D (Mpc)'].dropna()
distance_data = filtered_data[filtered_data < 4].values

# print(distance_data)

# Part (a) Plot histogram for number of bins = 10
print("------------------------------------")
print("Part (a) Plotting histogram with 10 bins")
bins = 10
hist, bin_edges = np.histogram(
    distance_data, bins=bins, range=(0, 4), density=False)

# Estimated probabilities for each bin
n = len(distance_data)
h = bin_edges[1] - bin_edges[0]
estimated_probabilities = hist / n

# Plot histogram
plt.figure()
plt.xlim(0, 4)
plt.xticks(np.arange(0, 4.1, 0.4))
plt.hist(distance_data, bins=bins, range=(0, 4), edgecolor='black')
plt.title('Histogram with 10 bins')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Frequency')
plt.savefig('../images/1/10binhistogram.png')

# Print estimated probabilities for each bin
print("Save the histogram as 10binhistogram.png")
print(f"Estimated probabilities (p_j) for each bin:")
# print each to exactly 3 decimal places
print(", ".join(f"{x:.3f}" for x in estimated_probabilities))
print("------------------------------------")
print("Part (b) Comment on the probability distribution")
print("The probability distribution is an underfit")
print("------------------------------------")
print("Part (c) Calculate cross-validation score for 1 to 1000 bins")
# Part (c) Calculate cross-validation score for bin widths from 1 to 1000 bins
cross_validation_scores = []
bin_range = range(1, 1001)

for m in bin_range:
    hist, bin_edges = np.histogram(
        distance_data, bins=m, range=(0, 4), density=False)
    v_j = hist
    p_j = v_j / n
    h = bin_edges[1] - bin_edges[0]

    term_1 = 2 / ((n - 1) * h)
    term_2 = (n + 1) / ((n - 1) * h)
    sum_p_j_squared = np.sum(p_j ** 2)

    cross_validation_score = term_1 - term_2 * sum_p_j_squared
    cross_validation_scores.append(cross_validation_score)

# Plot the cross-validation scores
plt.figure()
plt.plot(bin_range, cross_validation_scores, linewidth=0.75)
plt.title('Cross-validation scores vs. Bin Widths')
plt.xlabel('Number of bins')
plt.ylabel('Cross-validation score')
plt.savefig('../images/1/crossvalidation.png')
print("Saved the cross-validation scores vs number of bins graph as crossvalidation.png")

print("------------------------------------")
print("Part (d) Find the optimal value of h (bin width)")
# Part (d) Find the optimal value of h (bin width)
# +1 because range starts from 1
m_optimal = 1+np.argmin(cross_validation_scores)
h_optimal = 4 / m_optimal
print(f"Optimal number of bins: {m_optimal}")
print(f"Optimal bin width: {h_optimal:.3f}")

print("------------------------------------")
print("Part (e) Plot histogram with the optimal bin width and compare with the 10-bin histogram")
# Part (e) Plot histogram with the optimal bin width and compare with the 10-bin histogram
plt.figure()
plt.hist(distance_data, bins=m_optimal, range=(0, 4), edgecolor='black')
plt.xlim(0, 4)
plt.title(f'Optimal Histogram with {m_optimal} bins')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Frequency')
plt.savefig('../images/1/optimalhistogram.png')
print("Saved the optimal histogram as optimalhistogram.png")
print("------------------------------------")
