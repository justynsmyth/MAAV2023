import tempfile
import os
import shutil
import argparse
from Generator_v2 import imageGenerate
from definitions import shapes, color_options, symbols
from CharRecognize_v2 import charRecognize, testModel

'''
ArgParse config
'''
parser = argparse.ArgumentParser(
    description="A test script to test character recognition.")
parser.add_argument("-a", "--all", help="Run script on all combinations.",
                    action="store_true")  # changes args.all to true
parser.add_argument(
    "-t", "--train", help="generate responses and samples data", action="store_true")
parser.add_argument(
    "-r", "--run", action="store_true")
parser.add_argument("--shape", nargs='+', default=shapes,
                    help="Specify the shapes to select")
parser.add_argument("--symbol", nargs="+", default=symbols)
parser.add_argument("--color", nargs='+', default=color_options)


args = parser.parse_args()

if not any(vars(args).values()):
    parser.print_help()
    exit(1)


'''
class that initializes and deletes a temporary file after usage
'''


class TempDirectory(object):
    def __init__(self, dir_path):
        self.dir_path = tempfile.mkdtemp(dir=dir_path)

    def __del__(self):
        shutil.rmtree(self.dir_path)


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.abspath(__file__))

    temp_dir = TempDirectory(dir_path)

    print(temp_dir.dir_path)
    print('--------------------------')

    if args.train:
        print("Starting...")
        ctr = 0
        correct = 0
        for shape in shapes:
            for color in color_options.values():
                for symbol in symbols:
                    ctr += 1
                    filename = os.path.join(
                        temp_dir.dir_path, f'image{ctr}.png')
                    imageGenerate(
                        filename, 500, 500, color, shape, symbol)
                    charRecognize(filename, symbol)
        print("generated training files.")

    elif args.run:
        if args.all:
            print("generating ALL combinations...")
            ctr = 0
            for shape in shapes:
                for color in color_options.values():
                    for symbol in symbols:
                        ctr += 1
                        filename = os.path.join(
                            temp_dir.dir_path, f'image{ctr}.png')
                        imageGenerate(
                            filename, 500, 500, color, shape, symbol)
            print("finished generating")

        else:
            print("generating...")
            with open('output.txt', 'w') as file:
                file.write('')
            ctr = 0
            correct = 0
            for shape in args.shape:
                for color in args.color:
                    for symbol in args.symbol:
                        ctr += 1
                        file_path = os.path.join(
                            temp_dir.dir_path, f'image{ctr}.png')
                        imageGenerate(
                            file_path, 500, 500, color, shape, symbol)
                        print(file_path)
                        char = testModel(file_path)
                        if char == symbol:
                            correct += 1
                        else:
                            print("Error detected:")
                            with open('output.txt', 'a') as file:
                                file.write(
                                    f"Shape: {shape}, color: {color}, symbol: {symbol}\n")

                            # Define the name of the directory to be created
                            new_directory = 'errorImages'

                            # Define the path for the new directory
                            path = os.path.join(os.getcwd(), new_directory)

                            # Create the directory
                            try:
                                os.mkdir(path)
                                print(
                                    f"Directory '{new_directory}' created successfully.")
                            except FileExistsError:
                                print(
                                    f"Directory '{new_directory}' already exists.")
                            dest_path = os.path.join(
                                path, f'image{ctr}.png')
                            shutil.copyfile(file_path, dest_path)

            print("finished generating")
            print(f"correct {correct}")
            print(f"Accuracy: {correct / ctr }")

        # for filename in os.listdir(temp_dir.dir_path):
        #     file_path = os.path.join(temp_dir.dir_path, filename)
        #     print(file_path)
        #     testModel(file_path)
