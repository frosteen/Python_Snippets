import os
import time

from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import chromedriver_autoinstaller_fix


def url_contains(driver, url, timeout=300):
    for _ in range(timeout):
        if url in driver.current_url:
            return True
        time.sleep(1)
    return False


def find_xpath_element(driver, xpath, timeout=300):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element
    except StaleElementReferenceException:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element
    except TimeoutException:
        return False


def find_xpath_element_until_not(driver, xpath, timeout=300):
    try:
        element = WebDriverWait(driver, timeout).until_not(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element
    except StaleElementReferenceException:
        element = WebDriverWait(driver, timeout).until_not(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element
    except TimeoutException:
        return False


def find_xpath_elements(driver, xpath, timeout=300):
    try:
        elements = WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, xpath))
        )
        return elements
    except StaleElementReferenceException:
        elements = WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, xpath))
        )
        return elements
    except TimeoutException:
        return False


def find_xpath_elements_until_not(driver, xpath, timeout=300):
    try:
        elements = WebDriverWait(driver, timeout).until_not(
            EC.presence_of_all_elements_located((By.XPATH, xpath))
        )
        return elements
    except StaleElementReferenceException:
        elements = WebDriverWait(driver, timeout).until_not(
            EC.presence_of_all_elements_located((By.XPATH, xpath))
        )
        return elements
    except TimeoutException:
        return False


def find_xpath_element_button(driver, xpath, timeout=300):
    try:
        element_button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        return element_button
    except StaleElementReferenceException:
        element_button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        return element_button
    except TimeoutException:
        return False


def check_alert(driver, timeout=15):
    try:
        alert = (
            WebDriverWait(driver, timeout).until(EC.alert_is_present()).switch_to.alert
        )
        return alert
    except TimeoutException:
        return False


def until_clickable(element):
    while 1:
        try:
            element.click()
            break
        except:
            pass


def until_send_keys(element, text):
    while 1:
        try:
            element.send_keys(text)
            break
        except:
            pass


def scroll_check_click(driver, elem):
    for x in range(0, 51):
        try:
            height = int(
                int(driver.execute_script("return document.body.scrollHeight")) * x / 50
            )
            driver.execute_script(f"window.scrollTo(0, {height});")
            elem.click()
            break
        except:
            pass


def init_driver():
    chromedriver_autoinstaller_fix.install()
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument(
        rf"--user-data-dir=C:\Users\{os.getlogin()}\AppData\Local\Google\Chrome\User Data"
    )
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(300)
    return driver
