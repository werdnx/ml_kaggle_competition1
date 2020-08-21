import os
import shutil

walk_dir = '/home/werdn/input/bird/train_audio'
out_dir = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/output/bird/mfcc'


def main():
    shutil.copytree(walk_dir, out_dir)
    for root, subdirs, files in os.walk(out_dir):
        print('--\nroot = ' + root)
        list_file_path = os.path.join(root, 'my-directory-list.txt')
        print('list_file_path = ' + list_file_path)

        with open(list_file_path, 'wb') as list_file:
            for subdir in subdirs:
                print('\t- subdirectory ' + subdir)

            for filename in files:
                file_path = os.path.join(root, filename)
                parts = filename.split(".")
                # if parts[1] == 'mp3':
                if 'mfccs_m' in parts[0]:
                    print('do not remove file ' + filename)
                else:
                    print('\t- remove file %s (full path: %s)' % (filename, file_path))
                    os.remove(file_path)


if __name__ == "__main__":
    main()
