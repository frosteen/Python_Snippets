import pynmea2


def parse_gps(str):
    if str.find("GGA") > 0:
        try:
            msg = pynmea2.parse(str)
            return (
                msg.timestamp,
                msg.latitude,
                msg.longitude,
                msg.altitude,
                msg.altitude_units,
            )
        except Exception as e:
            pass
