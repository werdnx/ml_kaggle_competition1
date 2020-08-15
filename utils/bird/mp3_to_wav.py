import os

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

walk_dir = '/home/werdn/input/bird/train_audio'


def main():
    for root, subdirs, files in os.walk(walk_dir):
        print('--\nroot = ' + root)
        list_file_path = os.path.join(root, 'my-directory-list.txt')
        print('list_file_path = ' + list_file_path)

        with open(list_file_path, 'wb') as list_file:
            for subdir in subdirs:
                print('\t- subdirectory ' + subdir)

            for filename in files:
                file_path = os.path.join(root, filename)

                parts = filename.split(".")
                new_name = parts[0] + '.wav'
                print(parts)
                if parts[1] == 'mp3':
                    try:
                        print('\t- file %s (full path: %s)' % (filename, file_path))
                        sound = AudioSegment.from_mp3(file_path)
                        sound.export(os.path.join(root, new_name), format="wav")
                        os.remove(file_path)
                    except CouldntDecodeError:
                        print("{} is corrupted".format(file_path))


if __name__ == "__main__":
    main()
