# 生成程式

* https://www.promptingguide.ai/zh/applications/coding

## 範例

您是一位有用的代码助手，可以教初级开发人员如何编写代码。您选择的语言是Python。不解释代码，只生成代码块。

透過註解讓 LLM 寫程式

註解

```py
"""
1. 创建一个电影名称列表
2. 创建这些电影的评分列表
3. 组合他们并创建一个包含10部电影及其评级的json对象
"""
```

LLM 寫的程式

```py
movies = ["The Shawshank Redemption", "The Godfather", "The Dark Knight", "Schindler's List", "Pulp Fiction", "The Lord of the Rings: The Return of the King", "Forrest Gump", "Star Wars: Episode V - The Empire Strikes Back", "Inception", "The Silence of the Lambs"]
 
ratings = [9.3, 9.2, 9.0, 8.9, 8.9, 8.9, 8.8, 8.7, 8.7, 8.6]
 
movie_ratings = {}
 
for i in range(len(movies)):
    movie_ratings[movies[i]] = ratings[i]
 
json_object = json.dumps(movie_ratings, indent=4)
 
print(json_object)
```
