import sys
import csv
import netmiko
from concurrent.futures import ThreadPoolExecutor

"""
python SSH_Devices.py <username> <password>
"""


class MySSH:
    def __init__(self, ip, device_type, username=sys.argv[1], password=sys.argv[2]):
        self.conn_data = {
            "ip": ip,
            "device_type": device_type,
            "username": username,
            "password": password,
            "secret": password,
            "conn_timeout": 60,
            "auth_timeout": 60,
            "banner_timeout": 60,
            "read_timeout_override": 500,
        }
        self.conn = netmiko.ConnectHandler(**self.conn_data)
        self.handle_error(self.conn.enable())

    def handle_error(self, func):
        try:
            output = func
        except Exception as e:
            output = str(e)
        return output

    def disconnect(self):
        self.conn.disconnect()

    def send_command(self, command, **kwargs):
        output = self.handle_error(self.conn.send_command(command, **kwargs))
        return output

    def send_config_commands(self, commands: list, cmd_verify=False, **kwargs):
        output = self.handle_error(
            self.conn.send_config_set(commands, cmd_verify=cmd_verify, **kwargs)
        )
        return output

    def save_config(self):
        output = self.handle_error(self.conn.save_config())
        return output


def device(hostname, ip, device_type):
    OUTPUT = "---------------------------------------------------\n"
    try:
        access = MySSH(ip, device_type)
        OUTPUT += "HOSTNAME: " + hostname + "\n"
        OUTPUT += "IP_ADDRESS: " + ip + "\n"
        output = access.send_command("show ver")
        for line in output.split("\n"):
            if "version" in line.lower():
                ios_version = line[line.lower().find("version") :]
                break
        OUTPUT += ios_version + "\n"
        output = access.send_command("show inventory")
        for line in output.split("\n"):
            if "PID" in line:
                model = line[line.lower().find("pid") :].split(",")[0].strip()
                break
        OUTPUT += model + "\n"
        output = access.send_command("show inventory")
        for line in output.split("\n"):
            if "SN" in line:
                serial_number = line[line.lower().find("sn") :]
                break
        OUTPUT += serial_number + "\n"
        access.disconnect()
    except Exception as e:
        OUTPUT += str(e)
    print(OUTPUT)
    return OUTPUT


def main():
    list_of_devices = []
    with open("LAB_Devices.csv", "r") as file:
        csvreader = csv.DictReader(file)
        for x in csvreader:
            list_of_devices.append(x)

    threadpool_results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for x in list_of_devices:
            result = executor.submit(
                device, x["HOSTNAME"], x["IP_ADDRESS"], "cisco_ios"
            ).result()
            threadpool_results.append(result)

    with open("OUTPUTS.txt", "w") as file:
        for x in threadpool_results:
            file.write(x)


if __name__ == "__main__":
    main()
