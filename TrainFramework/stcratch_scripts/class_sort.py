class person(object):
    def __init__(self, name, age):
        self.__name = name
        self.__age = age
        self.old_class = None

    @property
    def name(self):
        return self.__name
    @property
    def age(self):
        return self.__age
if __name__ == "__main__":
    person_list = []


    person_list.append(person("aa", 23))
    person_list.append(person("bb", 46))
    person_list.append(person("cc", 77))

    person_list.append(person("aa", 12))
    person_list.append(person("bb", 56))
    person_list.append(person("cc", 34))
    person_list.append(person("dd", 68))

    person_list.append(person("aa", 15))
    person_list.append(person("bb", 86))
    person_list.append(person("cc", 39))
    person_list.append(person("dd", 28))
    person_list.append(person("ee", 99))

    for aperson in person_list:
        print(aperson.age)
    person_list.sort(key=lambda x:x.age)
    print("after:")
    for aperson in person_list:
        print(aperson.age)