import tempfile
import os
import shutil
import argparse
from Generator_v2 import imageGenerate
from definitions import shapes, color_options, symbols
from CharRecognize_v2 import trainModel, testModel
from CharRecognize_v3 import easyOCR

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
parser.add_argument(
    "-g", "--generate", action="store_true" 
)
parser.add_argument("--shape", nargs='+', default=shapes,
                    help="Specify the shapes to select.")
parser.add_argument("--symbol", nargs="+", default=symbols)
parser.add_argument("--color", nargs='+', default=color_options)
parser.add_argument("--charColor", nargs='+', default=color_options)


args = parser.parse_args()


def generateImages():
    new_root_directory =  'data'

    if not os.path.exists(new_root_directory):
        os.makedirs(new_root_directory)

    test_directory = os.path.join(new_root_directory, 'test')

    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    gt_filename = os.path.join(new_root_directory,  'gt.txt')
    with open(gt_filename, 'w') as file:
        pass


    ctr = 0
    for shape in shapes:
        for color in color_options:
            for symbol in symbols:
                for charColor in color_options:
                    if (charColor.capitalize() == color.capitalize()):
                        continue
                    ctr += 1
                    filename = os.path.join(
                        test_directory, f'image{ctr}.png')
                    imageGenerate(
                        filename, 500, 500, color, shape, symbol, charColor)
                    with open(gt_filename, 'a') as file:
                                file.write(
                                    f"test/image{ctr}.png\t{symbol}\n")
                    
def fixDirectory():
    # Define the name of the directory to be created
    new_directory = 'errorImages'

    # Define the path for the new directory
    path = os.path.join(os.getcwd(), new_directory)

    # Create the directory or clear it if it already exists
    try:
        os.mkdir(path)
        print(f"Directory '{new_directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{new_directory}' already exists. Clearing its contents.")
        # Remove all files in the directory
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def generateAndRecognize(train):
    fixDirectory()
    with open('output.txt', 'w') as file:
        pass
    ctr = 0
    correct = 0
    for shape in shapes:
        for color in color_options:
            for symbol in symbols:
                for charColor in color_options:
                    if (charColor.capitalize() == color.capitalize()):
                        continue
                    ctr += 1
                    filename = os.path.join(
                        temp_dir.dir_path, f'image{ctr}.png')
                    imageGenerate(
                        filename, 500, 500, color, shape, symbol, charColor)
                    if train:
                        trainModel(filename, symbol)
                    else:
                        # charGuess = testModel(filename)
                        charGuess = easyOCR(filename)
                        print(f"Guess: {charGuess}, Answer: {symbol}")
                        if charGuess == symbol:
                            correct += 1
                        else:
                            # creates a text file with all errors
                            print("Error detected:")
                            with open('output.txt', 'a') as file:
                                file.write(
                                    f"Shape: {shape}, color: {color}, symbol: {symbol}, charColor: {charColor}, CharGuess: {charGuess} \n")


                            # Define the path for the new directory
                            path = os.path.join(os.getcwd(), 'errorImages')
                            dest_path = os.path.join(
                                path, f'image{ctr}.png')
                            shutil.copyfile(filename, dest_path)
    print("finished generating")
    if not train:
        print(f"correct {correct}")
        print(f"Accuracy: {correct / ctr }")


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
        with open('generalsamples.data', 'w') as f:
            pass
        with open('generalresponses.data', 'w') as f:
            pass
        print("training...")
        generateAndRecognize(True)
        print("finished.")

    if args.generate:
        print("generating ALL combinations...")
        generateImages()
        print("finished generating")

    if args.all:
        print("generating ALL combinations...")
        generateAndRecognize(False)
        print("finished generating")

    elif args.run:
        fixDirectory()
        with open('output.txt', 'w') as file:
            pass
        print("generating...")
        with open('output.txt', 'w') as file:
            file.write('')
        ctr = 0
        correct = 0
        for shape in args.shape:
            for color in args.color:
                for symbol in args.symbol:
                    for charColor in args.charColor:
                        if (charColor.capitalize() == color.capitalize()):
                            continue
                        ctr += 1
                        file_path = os.path.join(
                            temp_dir.dir_path, f'image{ctr}.png')
                        imageGenerate(
                            file_path, 500, 500, color, shape, symbol, charColor)
                        print(file_path)
                        charGuess = easyOCR(file_path)
                        print(f"Guess: {charGuess}, Answer: {symbol}")
                        if charGuess == symbol.capitalize():
                            correct += 1
                        else:
                            # creates a text file with all errors
                            print("Error detected:")
                            with open('output.txt', 'a') as file:
                                file.write(
                                    f"Shape: {shape}, color: {color}, symbol: {symbol}, charColor: {charColor}, CharGuess: {charGuess} \n")


                            # Define the path for the new directory
                            path = os.path.join(os.getcwd(), 'errorImages')
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
