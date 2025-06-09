# Day 1

## 1. 环境准备

```python
# 在cmd中进行操作
# 查看python版本
python --version

# 如果cmd控制台显示了python的版本，则说明正确，否则需要进行环境变量的配置
```

## 2. Python入门

### 2.1 Python 变量，变量类型，作用域

- 基本变量的类型：int,float,double,bool,list

- 作用域: 全局作用域，局部作用域，使用global和nonlocal关键字

- 类型转换，可以使用int(),str()进行强制类型转换

  ```python
  # Excise 1
  name = "Alice"  # str
  age = 20        # int
  grades = [90, 85, 88]  # list
  info = {"name": "Alice", "age": 20}  # dict
  
  # 类型转换
  age_str = str(age)
  number = int("123")
  
  # 作用域
  x = 10  # 全局变量
  def my_function():
      y = 5  # 局部变量
      global x
      x += 1
      print(f"Inside function: x={x}, y={y}")
  
  my_function()
  print(f"Outside function: x={x}")
  
  ```

### 2.2  运算符

- 算术运算符 ：包括加减乘除，取余

- 比较运算符：=，!=, < , >

- 逻辑运算符: and, or, not

- 位运算符: <<, >>

  ```python
  # Excise 2
  a = 10
  b = 3
  print(a + b)  # 13
  print(a // b)  # 3（整除）
  print(a ** b)  # 1000（幂）
  
  # 逻辑运算
  x = True
  y = False
  print(x and y)  # False
  print(x or y)   # True
  
  # 比较运算
  print(a > b)  # True
     
  ```

  

### 2.3 语句：条件、循环、异常

- 条件语句：`if`, `elif`, `else`。

- 循环语句：`for`, `while`, `break`, `continue`。

- 异常处理：`try`, `except`, `finally`。

  ```python
  # Excise 3
  # 条件语句
  score = 85
  if score >= 90:
      print("A")
  elif score >= 60:
      print("Pass")
  else:
      print("Fail")
  
  # 循环语句
  for i in range(5):
      if i == 3:
          continue
      print(i)
  
  # 异常处理
  try:
      num = int(input("Enter a number: "))
      print(100 / num)
  except ZeroDivisionError:  # 除以0异常
      print("Cannot divide by zero!")
  except ValueError:
      print("Invalid input!")
  finally:
      print("Execution completed.")
  
  ```

### 2.4 函数：定义、参数、匿名函数、高阶函数

- 函数定义：`def`关键字，默认参数，可变参数（`args`, `*kwargs`）。

- 匿名函数：`lambda`。

- 高阶函数：接受函数作为参数或返回函数。

  ```python
  # Excise 4
  # 函数定义
  def greet(name, greeting="Hello"):
      return f"{greeting}, {name}!"
  
  print(greet("Alice"))  # Hello, Alice!
  print(greet("Bob", "Hi"))  # Hi, Bob!
  
  # 可变参数
  def sum_numbers(*args):
      return sum(args)
  print(sum_numbers(1, 2, 3, 4))  # 10
  
  # 匿名函数
  double = lambda x: x * 2
  print(double(5))  # 10
  
  # 高阶函数
  def apply_func(func, value):
      return func(value)
  print(apply_func(lambda x: x ** 2, 4))  # 16
  
  ```

### 2.5 包和模块：定义模块、导入模块、使用模块、第三方模块

- 模块：import语句，from ... import ...。

- 创建模块：一个.py文件。

- 包：包含__init__.py的文件夹。

- 第三方模块：如requests, numpy。

  ```python
  # EXcise 5
  # 创建模块 mymodule.py
  # mymodule.py
  def say_hello():
      return "Hello from module!"
  
  # 主程序
  import mymodule
  print(mymodule.say_hello())
  
  # 导入第三方模块
  import requests
  response = requests.get("https://api.github.com")
  print(response.status_code)  # 200
  
  # 包使用示例
  from mypackage import mymodule
  
  
  ```

  

### 2.6  类和对象

- 类定义：class关键字，属性和方法。

- 继承、多态、封装 (三大特性)

- 实例化对象

  ```python
  # Excise 6
  # 定义类
  class Student:
      def __init__(self, name, age):
          self.name = name
          self.age = age
  
      def introduce(self):
          return f"I am {self.name}, {self.age} years old."
  
  # 继承
  class GradStudent(Student):
      def __init__(self, name, age, major):
          super().__init__(name, age)
          self.major = major
  
      def introduce(self):
          return f"I am {self.name}, a {self.major} student."
  
  # 使用
  student = Student("Alice", 20)
  grad = GradStudent("Bob", 22, "CS")
  print(student.introduce())  # I am Alice, 20 years old.
  print(grad.introduce())     # I am Bob, a CS student.
  
  ```

  

### 2.7 装饰器

- 装饰器本质：高阶函数，接受函数并返回新函数。

- 使用`@`语法。

- 带参数的装饰器。

  ```python
  # Excise 7
  # 简单装饰器
  def my_decorator(func):
      def wrapper():
          print("Before function")
          func()
          print("After function")
      return wrapper
  
  @my_decorator
  def say_hello():
      print("Hello!")
  
  say_hello()
  
  # 带参数的装饰器
  def repeat(n):
      def decorator(func):
          def wrapper(*args, **kwargs):
              for _ in range(n):
                  func(*args, **kwargs)
          return wrapper
      return decorator
  
  @repeat(3)
  def greet(name):
      print(f"Hi, {name}!")
  
  greet("Alice")
  
  ```

  

### 2.8 文件操作

- 读写文本文件：open(), read(), write()。

- 上下文管理器：with语句。

- 处理CSV、JSON文件。

  ```python
  # Excise 8
  # 写文件
  with open("example.txt", "w") as f:
      f.write("Hello, Python!\n")
  
  # 读文件
  with open("example.txt", "r") as f:
      content = f.read()
      print(content)
  
  # 处理CSV
  import csv
  with open("data.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["Name", "Age"])
      writer.writerow(["Alice", 20])
  
  ```

  

### 2.9 git 相关操作

git init 初始化仓库

git add . 添加到本地暂存区

git commit -m “” 提交到本地仓库

git push origin main 推送到远程仓库

git config —global [user.name](http://user.name) “” 设置全局名称

git config —global user.email 设置全局提交邮箱
