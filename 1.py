class MyClass:
    def __init__(self):
        self.attribute = None


# Assuming attr_name is the string representing the attribute name
attr_name = "attribute"

# Creating an instance of MyClass
obj = MyClass()

# Using getattr to access the attribute dynamically
attr_value = getattr(obj, attr_name)

# Adding a value to the attribute
if attr_value is None:  # Check if the attribute is None (or some other initial value)
    setattr(obj, attr_name, "new_value")  # Setting the attribute to a new value

# Now obj.attribute has been set to "new_value"
print(obj.attribute)  # Output: new_value


d = {"apple": None}
d["apple"] = 3
d["aa"] = 4

print(d)
print(d.get("app", "numpy"))


d["apple"] = d["aa"]
print(d)
del d["aa"]
print(d)
