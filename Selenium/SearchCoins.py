import csv

from tabulate import tabulate

from MySelenium import *

driver = init_driver()
headers = [
    "COIN_NAME",
    "ADDRESS",
    "OWNABLE",
    "MARKETCAP",
    "IS_LOCKED",
    "BUY_TAX",
    "SELL_TAX",
    "HOLDERS",
    "TELEGRAM",
    "WEBSITE",
]
print("SEARCHING...")
with open("ShitCoins.csv", "r") as f:
    csv_reader = csv.reader(f)
    for line in csv_reader:
        data = []
        if not "0x" in line[0]:
            continue
        CONTRACT_ADDRESS = line[0]
        driver.get(f"https://www.dexanalyzer.io/token/{CONTRACT_ADDRESS}")
        COIN_NAME = find_xpath_element(driver, '//*[@id="token_name"]')
        OWNABLE = find_xpath_element(driver, '//*[@id="owner_status"]')
        MARKETCAP = find_xpath_element(driver, '//*[@id="marketcap_value"]/b')
        LP_LOCKED_STATUS = find_xpath_element(driver, '//*[@id="lock_status"]')
        BUY_TAX = find_xpath_element(driver, '//*[@id="buy_tax"]')
        SELL_TAX = find_xpath_element(driver, '//*[@id="sell_tax"]')
        HOLDERS = find_xpath_element(driver, '//*[@id="total_holder"]/a')
        TELEGRAM = find_xpath_element(driver, '//*[@id="telegram"]')
        WEBSITE = find_xpath_element(driver, '//*[@id="website"]')
        SCAM_RISK_LEVEL = find_xpath_element(driver, '//*[@id="scam_detector"]')
        if (
            COIN_NAME
            and OWNABLE
            and MARKETCAP
            and LP_LOCKED_STATUS
            and BUY_TAX
            and SELL_TAX
            and HOLDERS
            and TELEGRAM
            and WEBSITE
            and SCAM_RISK_LEVEL.text == "OK! DYOR ALWAYS."
        ):
            data.append(
                [
                    COIN_NAME.text,
                    CONTRACT_ADDRESS,
                    OWNABLE.text,
                    MARKETCAP.text,
                    LP_LOCKED_STATUS.text,
                    BUY_TAX.text,
                    SELL_TAX.text,
                    HOLDERS.text,
                    TELEGRAM.text,
                    WEBSITE.text,
                ]
            )
            print(tabulate(data, headers=headers, tablefmt="fancy_grid"))
print("DONE SEARCHING COINS!")
driver.close()
