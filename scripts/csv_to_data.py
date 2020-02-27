from common.helper import DataLocation
import os
import pandas

col_names = ['Image', 'Id']


def csv_format_into_folder(location_csv: str, location_folder: str, location_img: str):
    data = pandas.read_csv(location_csv, names=col_names)
    for index, row in data.iterrows():
        if index > 0:
            image, id = row[col_names[0]], row[col_names[1]]
            folder_name = location_folder + "/" + id
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            print(id)
            print(image)
            os.rename(location_img + "/" + image, folder_name + "/" + image)
        print(index)


def format_csv():
    csv_format_into_folder(location_csv="../" + DataLocation.WHALES.value + "/train.csv",
                           location_folder="../" + DataLocation.WHALES.value + "/train/",
                           location_img="../" + DataLocation.WHALES.value + "/train/train")


if __name__ == "__main__":
    format_csv()
