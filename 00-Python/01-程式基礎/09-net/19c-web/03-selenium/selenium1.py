from selenium import webdriver

driver = webdriver.Chrome()

def fbLogin(url,username, password):
    driver.get(url)
    # driver.find_element_by_id(usernameId).send_keys(username)
    driver.find_element("id", "email").send_keys(username)
    # driver.find_element_by_id(passwordId).send_keys(password)
    driver.find_element("id", "pass").send_keys(password)
    # driver.find_element_by_id(submit_buttonId).click()
    driver.find_element("name", "login").click()

myFbEmail = 'xxx@gmail.com'
myFbPassword = 'xxxxxxx'
fbLogin("https://www.facebook.com/", myFbEmail, myFbPassword)

