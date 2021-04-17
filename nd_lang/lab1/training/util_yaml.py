import yaml

# data = yaml.load(file_descriptor)
# yaml.dump(data)

def yaml_loader(filepath):
    """ Loads data from a yaml file """
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        return data


def yaml_dump(filepath, data):
    """ Writes data to a yaml file """
    with open(filepath, "w") as file_descriptor:
        yaml.dump(data, file_descriptor)


























