from googlesearch.googlesearch import GoogleSearch
response = GoogleSearch().search("ccc")
for result in response.results:
    print("Title: " + result.title)
    print("Content: " + result.getText())