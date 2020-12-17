import os


if __name__ == "__main__":
    folders = os.listdir("/home/thomasquirk/PycharmProjects/Bird Classifier/data/train")

    with open("labels.txt", "w") as f:
        for folder in folders:
            f.write(folder.title() + "\n")
