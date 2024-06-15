from MySelenium import *

driver = init_driver()
driver.get("https://www.google.com/search?q=test")
# Scrape Google Results
RESULTS = find_xpath_elements(driver, '//a//h3[@class="LC20lb MBeuO DKV0Md"]')
for x in RESULTS:
    print(x.text)
driver.close()
