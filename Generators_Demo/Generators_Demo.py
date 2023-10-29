# Using yield
def g_range_yield(a, b):
    counter = a
    while counter < b:
        yield counter
        counter = counter + 1


for x in g_range_yield(0, 10):
    print(x, end=" ")


print()


# Using generator comprehension
g_range_comprehension = (x for x in range(0, 10))

for x in g_range_comprehension:
    print(x, end=" ")
