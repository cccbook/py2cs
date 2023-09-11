# Search

* [SerpApi: Google Search API](https://serpapi.com/)
    * https://python.langchain.com/en/latest/modules/agents/tools/examples/serpapi.html

還沒設定好 SERPAPI_API_KEY 之前

```
$ python serpApi.py
Traceback (most recent call last):
  File "serpApi.py", line 3, in <module>
    search = SerpAPIWrapper()
  File "pydantic\main.py", line 342, in pydantic.main.BaseModel.__init__
pydantic.error_wrappers.ValidationError: 1 validation error for SerpAPIWrapper
__root__
  Did not find serpapi_api_key, please add an environment variable `SERPAPI_API_KEY` which contains it, or pass  `serpapi_api_key` as a named parameter. (type=value_error)
```

設定好 SERPAPI_API_KEY 之後

```
$ python serpApi.py
result= Barack Hussein Obama II
```


## 以下不知是否一定要安裝 (我有裝)

```
$ pip install google-search-results
Collecting google-search-results
  Downloading google_search_results-2.4.2.tar.gz (18 kB)
Requirement already satisfied: requests in c:\users\user\appdata\local\programs\python\python38\lib\site-packages (from google-search-results) (2.28.1)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\user\appdata\local\programs\python\python38\lib\site-packages (from requests->google-search-results) (1.26.13)
Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\user\appdata\local\programs\python\python38\lib\site-packages (from requests->google-search-results) (2.1.1)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\user\appdata\local\programs\python\python38\lib\site-packages (from requests->google-search-results) (2022.12.7)
Requirement already satisfied: idna<4,>=2.5 in c:\users\user\appdata\local\programs\python\python38\lib\site-packages (from requests->google-search-results) (3.4)
Building wheels for collected packages: google-search-results
  Building wheel for google-search-results (setup.py) ... done
  Created wheel for google-search-results: filename=google_search_results-2.4.2-py3-none-any.whl size=32077
sha256=544a99e0db8bf1cdc573b9fed4ae1649b1b264123accd81ff35da9fbd50f2901
  Stored in directory: c:\users\user\appdata\local\pip\cache\wheels\29\75\71\9bf68178a74593837f73b6e9d9a070d45d308bddfd2e95290a
Successfully built google-search-results
Installing collected packages: google-search-results
Successfully installed google-search-results-2.4.2

```
