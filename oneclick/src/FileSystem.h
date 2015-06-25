#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <iostream>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>

#if defined WIN32 || defined _WIN32
#include "dirent.h"
#elif defined __linux__
#include <dirent.h>
#else
#error "not implemeted for this platform"
#endif

// Simple wrapper for dirent.h
// Can easily iterate over files in given folder. Works on Linux too.
// In NTFS files are iterated over alphabetically, exFAT - almost random

namespace fs {
    class directory_iterator {
    private:
        DIR * dir = NULL;
        dirent * item = NULL;
        std::string parent;
        std::string item_path;
    public:
        /** Default constructor.
          */
        directory_iterator() {}

        /** Constructor, opens directory for reading, files inside can be accessed.
          * After creating points to first file in opened directory.
          * @param p        path to directory
          */
        directory_iterator(const std::string & p) {
            dir = opendir(p.c_str());
            if(dir != NULL) {
                item = readdir(dir);
                parent = p;
                item_path = parent + "/" + item->d_name;
            }
        }

        /** Destructor, closes directory.
          */
        ~directory_iterator() {
            closedir(dir);
        }

        /** Returns path to file directory_iterator is pointing at.
          */
        std::string path() { return item_path; }

        /** Returns name of file directory_iterator is pointing at or empty string.
          */
        std::string name() {
            if(item != NULL) {
                return (std::string)item->d_name;
            } else {
                return "";
            }
        }

        /** Moves pointer to next file in directory.
          */
        void operator++(int) {
            if(dir != NULL) {
                if(item == NULL) {
                    rewinddir(dir);
                    item = readdir(dir);
                } else {
                    item = readdir(dir);
                }
                item_path.erase();
                if(item != NULL) {
                    item_path = parent + "/" + item->d_name;
                } else {
                    item_path = "";
                }
            }
        }

        /** Equality operator, compares items of iterators.
          */
        bool operator==(const directory_iterator & b) { return (item == b.item); }

        /** Non - equality operator, compares items of iterators.
          */
        bool operator!=(const directory_iterator & b) { return (item != b.item); }

        /** Returns true if file poited at by iterator is directory.
          */
        bool is_directory() {
            struct stat s;
            stat(item_path.c_str(), &s);
            return S_ISDIR(s.st_mode);
        }
        /** Returns true if file poited at by iterator is regular file.
          */
        bool is_file() {
            struct stat s;
            stat(item_path.c_str(), &s);
            return S_ISREG(s.st_mode);
        }
    };
}

#endif //FILESYSTEM_H
