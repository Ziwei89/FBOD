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

class family(object):
    def __init__(self,fname):
        self.__fname = fname
        self.persons = []
    
    @property
    def fname(self):
        return self.__fname
if __name__ == "__main__":
    family1 = family("zhang")
    family2 = family("wang")
    family3 = family("sun")

    family1.persons.append(person("aa", 23))
    family1.persons.append(person("bb", 46))
    family1.persons.append(person("cc", 77))

    family2.persons.append(person("aa", 12))
    family2.persons.append(person("bb", 56))
    family2.persons.append(person("cc", 34))
    family2.persons.append(person("dd", 68))

    family3.persons.append(person("aa", 15))
    family3.persons.append(person("bb", 86))
    family3.persons.append(person("cc", 39))
    family3.persons.append(person("dd", 28))
    family3.persons.append(person("ee", 99))

    age_list = []
    family_list = [family1,family2,family3]

    for familyi in family_list:
        print(len(familyi.persons))
        for n in range(len(familyi.persons)):
            age_list.append(familyi.persons[n].age)
    print(age_list)
    age_list.sort()
    print(age_list)
    print(age_list[3-1], age_list[6-1], age_list[9-1], age_list[12-1])

    for familyi in family_list:
        for n in range(len(familyi.persons)):
            if familyi.persons[n].age <= age_list[3-1]:
                familyi.persons[n].old_class = "Younger"
            elif familyi.persons[n].age > age_list[3-1] and familyi.persons[n].age <= age_list[6-1]:
                familyi.persons[n].old_class = "Yonug"
            elif familyi.persons[n].age > age_list[6-1] and familyi.persons[n].age <= age_list[9-1]:
                familyi.persons[n].old_class = "old"
            else:
                familyi.persons[n].old_class = "older"
    for familyi in family_list:
        print(familyi.fname, "family: ")
        for n in range(len(familyi.persons)):
            print(familyi.persons[n].name, familyi.persons[n].age, familyi.persons[n].old_class)