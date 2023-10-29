import matplotlib.pyplot as plt

x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.title("Title")

plt.xlabel("x-axis")

plt.ylabel("y-axis")

plt.tight_layout()

plt.xticks(rotation=45)

plt.plot(x_values)
plt.plot(x_values, y_values)

plt.show()
