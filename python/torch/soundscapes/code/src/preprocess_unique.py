import os
import sys
from shutil import copyfile

import audio_metadata


def main():
    src = sys.argv[1]
    target = sys.argv[2]
    flacs = [(os.path.join(src, i), i) for i in os.listdir(src)]
    dict = {}
    unique = 0
    total = 0
    for flac in flacs:
        s = flac[1].split('.')
        if s[1] == 'flac':
            total += 1
            metadata = audio_metadata.load(flac[0])
            if metadata.streaminfo.md5 in dict:
                dict[metadata.streaminfo.md5][1] += 1
            else:
                dict[metadata.streaminfo.md5] = [flac, 1]
    for k in dict:
        if dict[k][1] == 1:
            copyfile(dict[k][0][0], os.path.join(target, dict[k][0][1]))
            print('file ' + dict[k][0][0] + ' copied to ' + target + dict[k][0][1])
            unique += 1

    print('total ' + str(total))
    print('unique ' + str(unique))


if __name__ == "__main__":
    main()
