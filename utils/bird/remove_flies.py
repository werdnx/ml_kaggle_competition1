import os

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

                print('\t- file %s (full path: %s)' % (filename, file_path))
                parts = filename.split(".")
                if parts[1] == 'npy':
                    os.remove(file_path)
                # with open(file_path, 'rb') as f:
                #     f_content = f.read()
                #     list_file.write(('The file %s contains:\n' % filename).encode('utf-8'))
                #     list_file.write(f_content)
                #     list_file.write(b'\n')


if __name__ == "__main__":
    main()
