import time


class Count:
    def __init__(self, class_name="cherry", time_threshold=3):
        """
        Avoid overlapping the total by
        checking the previous frame
        if total is still the same :)
        """
        self.class_name = class_name
        self.time_threshold = time_threshold  # check every 3 seconds
        self.last_count_time = 0
        self.prev_count = 0
        self.count = 0
        self.total = 0
        self.time_now = 0

    def update_total(self, object_names):
        """ "
        object_names accepts only dataframe.name.value_counts()
        """

        self.time_now = time.time()

        if self.class_name in object_names:

            self.count = object_names[self.class_name]

            if self.count > self.prev_count:

                if (self.time_now - self.last_count_time) > self.time_threshold:

                    new_count = self.count - self.prev_count

                    self.total = self.total + new_count

                    self.last_count_time = self.time_now

                else:

                    self.last_count_time = self.time_now

        elif self.class_name not in object_names:

            self.count = 0

        self.prev_count = self.count
