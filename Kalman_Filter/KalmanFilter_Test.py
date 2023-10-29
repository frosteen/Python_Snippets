from KalmanFilter import KalmanFilter
from matplotlib import pyplot

test = KalmanFilter(0.008, 0.1)
test_data = [66, 64, 63, 63, 63, 66, 65, 67, 58]
output_data = []

for x in test_data:
    print("Data:", x)
    filtered_data = test.filter(x)
    output_data.append(filtered_data)
    print("Filtered Data:", filtered_data)

pyplot.plot(test_data)
pyplot.plot(output_data)
pyplot.show()
