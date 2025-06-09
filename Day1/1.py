x = 10  # 全局变量
def my_function():
    y = 5  # 局部变量
    global x
    x += 1
    print(f"Inside function: x={x}, y={y}")

my_function()
print(f"Outside function: x={x}")