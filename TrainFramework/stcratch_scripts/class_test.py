class person(object):
    def __init__(self, name, age):
        self.__name = name
        self.__age = age
        self.grade=5

    @property
    def name(self):
        return self.__name
    @property
    def age(self):
        temp_age = self.__age
        self.__age = None
        return temp_age
    @age.setter
    def age(self, age):
        self.__age=age

sun = person("sun zi wei",35)
print(sun.name)
print(sun.age)
sun.age=34
print(sun.age)
print(sun.age)