from concurrent.futures import ThreadPoolExecutor
import time


def sum(a, b):
    ans = a + b
    time.sleep(1)
    return ans


def main():
    start_time = time.time()
    #################################################
    numbers = [
        (1, 2),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]
    threadpool_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for x in numbers:
            result = executor.submit(sum, x[0], x[1]).result()
            threadpool_results.append(result)
    for x in threadpool_results:
        print("SUM:", x)
    #################################################
    end_time = time.time()
    print("Elapsed Time:", end_time - start_time)


if __name__ == "__main__":
    main()
