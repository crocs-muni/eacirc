#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <iostream>
#include <string>

#if defined WIN32 || defined _WIN32
#include "dirent.h"
#elif defined __linux__
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#else
#error "not implemeted for this platform"
#endif

// Simple wrapper for dirent.h

namespace fs {
	class directory_iterator {
	private:
		DIR * dir = NULL;
		dirent * item = NULL;
		std::string parent;
		std::string item_path;
	public:
		directory_iterator() {}

		directory_iterator(std::string p) {
			dir = opendir(p.c_str());
			if(dir != NULL) {
				item = readdir(dir);
				parent = p;
				item_path = parent + "/" + item->d_name;
			}
		}

		~directory_iterator() {
			closedir(dir);
		}

		std::string path() { return item_path; }

		std::string name() {
			if(item != NULL) {
				return (std::string)item->d_name;
			} else {
				return "";
			}
		}

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

		bool operator==(directory_iterator b) { return (item == b.item); }

		bool operator!=(directory_iterator b) { return (item != b.item); }

		bool is_directory() { return S_ISDIR(item->d_type); }

		bool is_file() { return S_ISREG(item->d_type); }
	};
}









#endif //FILESYSTEM_H
