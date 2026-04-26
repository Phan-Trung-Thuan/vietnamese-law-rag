# Suitable for Google Colab, for local please follow the external instructions and ignore this line
# and follows https://docs.google.com/document/d/14jK9d6KHJYX0b-gFAVqAghUxT7OLAM0nP2IovL7_Rjs/edit?usp=sharing
# !apt install -qq chromium-chromedriver

# Install selenium
# !pip install -qq selenium

# Tạo thư mục để chứa data
# !mkdir -p data

# selenium import
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
# other imports
import os
import json

# selenium setups
## https://www.tutorialspoint.com/selenium/selenium_webdriver_chrome_webdriver_options.htm

chrome_options = webdriver.ChromeOptions()

chrome_options.add_argument('--headless') # must options for Google Colab
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")


# HOME_PAGE = "https://thuvienphapluat.vn/hoi-dap-phap-luat/tim-tu-van?searchType=1&q=sinh+vi%C3%AAn&searchField=22%40page%3d2"
# HOME_PAGE = "https://thuvienphapluat.vn/hoi-dap-phap-luat/giao-duc?"
HOME_PAGE = "https://thuvienphapluat.vn/hoi-dap-phap-luat/tim-tu-van?searchType=1&q=%C4%91%E1%BA%A1i+h%E1%BB%8Dc&searchField=22"

page_subfix = '&page='
a = 1
b = a + 20
page_id = [i for i in range(a, b+1)]

driver = webdriver.Chrome(options=chrome_options)

page_links = []
keyword_set = set()
for id in page_id:
    url = HOME_PAGE + page_subfix + str(id)
    driver.get(url)
    driver.implicitly_wait(5)

    news_cards = driver.find_elements(by=By.CLASS_NAME, value="news-card")
    print(f'Page {id} has {len(news_cards)} elements')

    for i, card in enumerate(news_cards):
        try:
            keyword = card.find_element(by=By.CLASS_NAME, value="sub-item-head-keyword")
            keyword_set.add(keyword.text)
        except:
            continue

driver.close()

print(*keyword_set, sep='\n')

keywords = keyword_set.union(set(['Điểm chuẩn đại học', 'Sinh viên', 'Sinh hoạt công dân', 'Thi tốt nghiệp THPT',
            'Tuyển sinh Đại học', 'Giáo dục đại học']))
print(len(keywords), len(keyword_set))

page_subfix = '&page='
a = 1
b = a + 50
page_id = [i for i in range(a, b+1)]

driver = webdriver.Chrome(options=chrome_options)

page_links = []
for id in page_id:
    url = HOME_PAGE + page_subfix + str(id)
    driver.get(url)
    driver.implicitly_wait(5)

    news_cards = driver.find_elements(by=By.CLASS_NAME, value="news-card")
    print(f'Page {id} has {len(news_cards)} elements')

    for card in news_cards:
        try:
            keyword = card.find_element(by=By.CLASS_NAME, value="sub-item-head-keyword")
            if keyword.text in keywords:
                link_element = card.find_element(By.TAG_NAME, "a")
                href_link = link_element.get_attribute("href")
                page_links.append(href_link)
        except:
            link_element = card.find_element(By.TAG_NAME, "a")
            href_link = link_element.get_attribute("href")
            page_links.append(href_link)

driver.close()

print(len(page_links))
# print(*page_links, sep='\n')

def preprocess_answer(list_text):
    tmp = []
    for text in list_text:
        if text != '' and 'Hình' not in text and '?' not in text and '\n' not in text:
            tmp.append(text)
    return tmp

answer_spliters = ('nếu', 'như vậy', 'theo đó', 'theo quy định nêu trên', 'căn cứ quy định nêu trên',
                  'tùy thuộc vào', 'theo như quy định nêu trên', 'căn cứ vào', 'do đó', 'trên đây là',
                   'theo các quy định trên', 'căn cứ theo quy định hiện hành', 'theo quy định này',
                   'theo quy định trên')
answer_keywords = ('nêu trên', 'nguyên tắc trên', 'quy định như trên', 'quy định nêu trên',
                   'đã trình bày ở trên', 'đã trình bày bên trên', 'quy định pháp luật')
wrong_words = ('điều kiện', 'mục tiêu')
A = ('tại', 'theo', 'căn cứ', 'căn cứ tại', 'căn cứ theo')
B = ('điều', 'mục', 'khoản', 'quy định', 'điểm') # Điều luật, mục lục trong luật
document_spliters = tuple(a + ' ' + b for a in A for b in B) + B
wrong_spliters = tuple(a + ' ' + b for a in A for b in wrong_words) + wrong_words
# print(*document_spliters, sep='\n')

def split_documents_answer(list_text):
    answer = []
    documents = []

    for i in range(len(list_text)):
        if any(list_text[i].lower().startswith(spliter) for spliter in answer_spliters) or any(keyword in list_text[i].lower() for keyword in answer_keywords):
            answer = list_text[i:]
            documents = list_text[:i]
            break

    if len(answer) == 0:
        documents = list_text

    if len(documents) == 0:
        return None, None

    print(len(documents), len(answer))

    split_index = [0]
    for i in range(1, len(documents)):
        if any(documents[i].lower().startswith(spliter) for spliter in document_spliters) and not any(documents[i].lower().startswith(spliter) for spliter in wrong_spliters):
            split_index.append(i)
    split_index.append(len(documents))

    documents = [{
        'name': documents[split_index[i]],
        'law': '\n'.join(documents[split_index[i]+1:split_index[i+1]])
    } for i in range(len(split_index) - 1)]

    return documents, answer

driver = webdriver.Chrome(options=chrome_options)

data_dict = []

for i, link in enumerate(page_links):
    print(i, link)
    driver.get(link)
    driver.implicitly_wait(2)

    questions = driver.find_elements(By.CSS_SELECTOR, 'strong[style="font-size: 12pt;"]')
    page_type = 1
    if len(questions) == 0:
        questions = driver.find_elements(By.CSS_SELECTOR, "[id^='mucluc-']")
        page_type = 2

    if len(questions) == 0:
        questions = driver.find_elements(By.CLASS_NAME, 'sapo')
        page_type = 3

    if page_type == 1:
        questions = questions[len(questions)//2:]

    if len(questions) == 0:
        continue

    # Step 2: Loop over all questions and find the answer between them
    for i in range(len(questions) - 1):
        if '?' not in questions[i].text:
            continue

        question1 = questions[i]
        question2 = questions[i+1]

        if page_type == 1:
            question1 = question1.find_element(By.XPATH, '..')
            question2 = question2.find_element(By.XPATH, '..')

        current_answer = []

        try:
            next_element = question1.find_element(By.XPATH, "following-sibling::*[1]")
            while next_element != question2:
                if page_type == 1:
                    current_answer.append(next_element.text)
                else:
                    current_answer.extend(next_element.text.split('\n'))

                # Try to move to the next sibling
                try:
                    next_element = next_element.find_element(By.XPATH, "following-sibling::*[1]")
                except:
                    break  # Stop if there are no more siblings

            current_answer = preprocess_answer(current_answer)
            if len(current_answer) == 0:
                continue

            documents, answer = split_documents_answer(current_answer)
            if documents is None:
                continue

            data_dict.append({
                'link': link,
                'question': questions[i].text,
                'documents': documents,
                'answer': '\n'.join(answer)
            })
        except:
            continue

    # Get last answer
    try:
        current_answer = []
        question1 = questions[-1]
        if page_type == 1:
            question1 = question1.find_element(By.XPATH, '..')

        next_element = question1.find_element(By.XPATH, "following-sibling::*[1]")
        while True:
            if page_type == 1:
                current_answer.append(next_element.text)
            else:
                current_answer.extend(next_element.text.split('\n'))

            try:
                next_element = next_element.find_element(By.XPATH, "following-sibling::*[1]")
            except:
                break  # Stop if there are no more siblings

        current_answer = preprocess_answer(current_answer)
        if len(current_answer) == 0:
            continue

        documents, answer = split_documents_answer(current_answer)
        if documents is None:
            continue

        data_dict.append({
            'link': link,
            'question': questions[-1].text,
            'documents': documents,
            'answer': '\n'.join(answer)
        })
    except:
        continue

driver.close()

print(len(data_dict))

print(json.dumps(data_dict, indent=4, ensure_ascii=False))

# For Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Set to your folder
FOLDER_SAVED_GOOGLE_COLAB = "/content/drive/MyDrive/"

with open(FOLDER_SAVED_GOOGLE_COLAB + '/data_dict.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)

